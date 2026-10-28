"""Build the committed HI-Small sample from the raw AMLworld files.

This developer tool is the only code that reads the raw transactions CSV, the
accounts CSV and the Patterns file. The sample keeps the original file names,
header lines and row text, so the same loader reads the sample and the full
data. The output depends only on the raw files, the seed and the row target, so
a rerun writes byte-identical files.

Run it from the repository root with `make sample`.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from sage_xgb.config import SplitConfig, load_config
from sage_xgb.split import add_day_and_split


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
HI_SMALL_CONFIG = REPO_ROOT / "configs" / "hi_small.yaml"
DEFAULT_OUT_DIR = Path("tests/fixtures/amlworld_hi_small_sample")
DEFAULT_SEED = 7
DEFAULT_ROWS = 28_000
KAGGLE_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

TRANS_FILE = "HI-Small_Trans.csv"
ACCOUNTS_FILE = "HI-Small_accounts.csv"
PATTERNS_FILE = "HI-Small_Patterns.txt"
TRANS_COLUMNS = (
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
)
ACCOUNT_COLUMNS = (
    "bank_name",
    "bank_id",
    "account_number",
    "entity_id",
    "entity_name",
)

TYPOLOGIES = (
    "FAN-OUT",
    "FAN-IN",
    "GATHER-SCATTER",
    "SCATTER-GATHER",
    "CYCLE",
    "BIPARTITE",
    "STACK",
    "RANDOM",
)
PATTERNS_PER_TYPOLOGY = 2
DAY_BLOCKS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))
BLOCK_LABELS = (*(f"{first}-{last}" for first, last in DAY_BLOCKS), "tail")
MIN_ILLICIT_PER_BLOCK = 10
MIN_ILLICIT_TAIL = 3
TAIL_ROWS = 20
GAP_MINUTES = (60, 1440, 10080)
NEIGHBOUR_WINDOW_MINUTES = 1440
PICK_POOL_SIZE = 2000
BEGIN_PREFIX = "BEGIN LAUNDERING ATTEMPT - "
END_PREFIX = "END LAUNDERING ATTEMPT - "


@dataclass(frozen=True)
class Pattern:
    """One laundering attempt from the Patterns file.

    Attributes:
        index: Position among the patterns in the file, from 0.
        typology: Typology name such as FAN-OUT.
        lines: Raw lines from the BEGIN line through the END line.
        transactions: Text fields of each transaction line, in file order.
    """

    index: int
    typology: str
    lines: tuple[str, ...]
    transactions: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _PatternSummary:
    """Matched size and illicit row counts of one pattern."""

    index: int
    typology: str
    length: int
    block_counts: Counter[str]


@dataclass(frozen=True)
class _Candidate:
    """An edge case that the sample must contain, with its neighbourhood.

    Attributes:
        rows: Row ids that carry the edge case, used as a stable tiebreak.
        involved: Account keys whose Transactions are kept with the case.
        account: Account whose stream needs a neighbour, if any.
        minute: Minute of the later Transaction, if the neighbour check applies.
    """

    rows: tuple[int, ...]
    involved: tuple[str, ...]
    account: str | None = None
    minute: int | None = None


def parse_patterns(text: str) -> list[Pattern]:
    """Parse the Patterns file into one entry per laundering attempt.

    Args:
        text: Full text of the Patterns file.

    Returns:
        The patterns in file order. Blank lines outside a block are ignored.

    Raises:
        RuntimeError: A block is not closed, mixes typologies, names an unknown
            typology, or holds a line that is not a transaction.
    """
    patterns: list[Pattern] = []
    typology = ""
    span: list[str] = []
    transactions: list[tuple[str, ...]] = []
    for line in text.split("\n"):
        if line.startswith(BEGIN_PREFIX):
            _require(not span, f"a pattern starts before the last one ends: {line}")
            typology = _block_typology(line, BEGIN_PREFIX)
            span = [line]
            transactions = []
        elif not span:
            _require(line == "", f"line outside a pattern block: {line!r}")
        elif line.startswith(END_PREFIX):
            _require(
                _block_typology(line, END_PREFIX) == typology,
                f"END line does not match its BEGIN line: {line}",
            )
            span.append(line)
            patterns.append(
                Pattern(
                    index=len(patterns),
                    typology=typology,
                    lines=tuple(span),
                    transactions=tuple(transactions),
                )
            )
            span = []
        else:
            span.append(line)
            if line:
                fields = tuple(line.split(","))
                _require(
                    len(fields) == len(TRANS_COLUMNS),
                    f"transaction line has {len(fields)} fields: {line!r}",
                )
                transactions.append(fields)
    _require(not span, "the Patterns file ends inside a pattern block")
    return patterns


def build_sample(*, data_dir: Path, out_dir: Path, seed: int, rows: int) -> None:
    """Write the HI-Small sample files to `out_dir`.

    Args:
        data_dir: Directory that holds the three raw HI-Small files.
        out_dir: Directory for the sample. Its sample files are replaced.
        seed: Seed for the Tail and background draws.
        rows: Target row count of the transactions file. The pattern closure
            and the edge-case neighbourhoods come first, and the background
            fills the rest.

    Raises:
        RuntimeError: The raw files do not match the expected layout, or the
            selected sample misses a property the tests rely on.
    """
    split = load_config(HI_SMALL_CONFIG).split
    trans_path = data_dir / TRANS_FILE
    light = _load_transactions(trans_path, split)
    logger.info(
        "Read %d transactions, %d illicit.",
        light.height,
        light["illicit"].sum(),
    )
    _log_padded_bank_ids(light)

    patterns = parse_patterns((data_dir / PATTERNS_FILE).read_text(encoding="utf-8"))
    matches = _match_patterns(trans_path, patterns)
    summaries, unmatched, partial = _summarise_patterns(patterns, matches, light)
    logger.info(
        "Patterns: %d complete, %d partly matched, %d matched nothing.",
        len(summaries),
        len(partial),
        len(unmatched),
    )
    logger.info("Patterns that matched nothing: %s", unmatched)
    logger.info("Patterns with unmatched lines: %s", partial)

    chosen = _choose_patterns(summaries)
    logger.info(
        "Chose %d patterns: %s",
        len(chosen),
        [(s.index, s.typology, s.length) for s in chosen],
    )
    pattern_rows = (
        matches.filter(pl.col("pattern").is_in([s.index for s in chosen]))["row_id"]
        .unique()
        .sort()
        .to_list()
    )
    pattern_accounts = _accounts_of(light, pattern_rows)
    closure = light.filter(
        pl.col("src").is_in(pattern_accounts) & pl.col("dst").is_in(pattern_accounts)
    )["row_id"].to_list()
    logger.info(
        "Pattern rows: %d. Pattern closure: %d rows.",
        len(pattern_rows),
        len(closure),
    )

    edge_rows = _edge_case_rows(light)
    rng = np.random.default_rng(seed)
    tail_rows = _tail_rows(light, rng)
    core = set(pattern_rows) | set(closure) | edge_rows | set(tail_rows)
    _require(
        len(core) <= rows,
        f"pattern closure and edge cases need {len(core)} rows, above the "
        f"target of {rows}; choose fewer or smaller patterns",
    )
    background = _background_rows(light, core, rows - len(core), rng)
    kept = sorted(core | set(background))
    logger.info("Kept %d rows in total.", len(kept))

    kept_light = light.filter(pl.col("row_id").is_in(kept))
    _verify_sample(
        kept_light,
        matches=matches,
        chosen_indices={s.index for s in chosen},
        pattern_rows=pattern_rows,
        closure=closure,
    )

    account_row_ids = _account_line_ids(data_dir / ACCOUNTS_FILE, kept_light)
    out_dir.mkdir(parents=True, exist_ok=True)
    trans_lines = _copy_selected_lines(trans_path, out_dir / TRANS_FILE, set(kept))
    _require(
        trans_lines == light.height,
        f"{TRANS_FILE} has {trans_lines} data lines but {light.height} rows parsed",
    )
    accounts_lines = _copy_selected_lines(
        data_dir / ACCOUNTS_FILE, out_dir / ACCOUNTS_FILE, set(account_row_ids)
    )
    accounts_rows = (
        _scan_raw(data_dir / ACCOUNTS_FILE, ACCOUNT_COLUMNS)
        .select(pl.len())
        .collect()
        .item()
    )
    _require(
        accounts_lines == accounts_rows,
        f"{ACCOUNTS_FILE} has {accounts_lines} data lines but {accounts_rows} rows",
    )
    logger.info("Kept %d of %d account lines.", len(account_row_ids), accounts_lines)
    chosen_patterns = [patterns[s.index] for s in chosen]
    (out_dir / PATTERNS_FILE).write_text(
        "".join(f"{line}\n" for pattern in chosen_patterns for line in pattern.lines),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        _readme(
            kept_light=kept_light,
            seed=seed,
            rows=rows,
            pattern_count=len(chosen_patterns),
        ),
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    """Raise a RuntimeError with `message` unless `condition` holds."""
    if not condition:
        raise RuntimeError(message)


def _block_typology(line: str, prefix: str) -> str:
    """Return the typology named after `prefix` on a BEGIN or END line."""
    name = line.removeprefix(prefix).split(":", 1)[0].strip()
    _require(name in TYPOLOGIES, f"unknown typology on line: {line}")
    return name


def _with_newline(line: bytes) -> bytes:
    """Return `line` with a trailing newline."""
    return line if line.endswith(b"\n") else line + b"\n"


def _scan_raw(path: Path, columns: Sequence[str]) -> pl.LazyFrame:
    """Scan a headed CSV with every value as text and a 0-based `row_id`.

    The header line is skipped. Keeping values as text preserves leading zeros
    in bank ids and the exact amount text used for matching.
    """
    return pl.scan_csv(
        path,
        has_header=False,
        skip_rows=1,
        new_columns=list(columns),
        infer_schema=False,
    ).with_row_index("row_id")


def _block_expr() -> pl.Expr:
    """Return the day block label of each row: a block name, or tail."""
    expr = pl.when(pl.col("split") == "tail").then(pl.lit("tail"))
    for first, last in DAY_BLOCKS:
        expr = expr.when(pl.col("day").is_between(first, last)).then(
            pl.lit(f"{first}-{last}")
        )
    return expr.otherwise(pl.lit(None, dtype=pl.String))


def _load_transactions(trans_path: Path, split: SplitConfig) -> pl.DataFrame:
    """Load the keys, split, day block and label of every Transaction.

    Args:
        trans_path: The raw transactions CSV.
        split: The split definition from the HI-Small config.

    Returns:
        One row per data row, sorted by `row_id`.
    """
    frame = _scan_raw(trans_path, TRANS_COLUMNS).with_columns(
        timestamp=pl.col("timestamp").str.to_datetime("%Y/%m/%d %H:%M")
    )
    frame = add_day_and_split(frame, split).with_columns(
        block=_block_expr(),
        src=pl.concat_str(["from_bank", "from_account"], separator=":"),
        dst=pl.concat_str(["to_bank", "to_account"], separator=":"),
        minute=pl.col("timestamp").dt.epoch("s") // 60,
        illicit=pl.col("is_laundering") == "1",
    )
    return (
        frame.select(
            "row_id",
            "day",
            "split",
            "block",
            "from_bank",
            "from_account",
            "to_bank",
            "to_account",
            "src",
            "dst",
            "minute",
            "illicit",
        )
        .sort("row_id")
        .collect()
    )


def _match_patterns(trans_path: Path, patterns: Sequence[Pattern]) -> pl.DataFrame:
    """Find the raw rows that equal each pattern transaction line.

    Args:
        trans_path: The raw transactions CSV.
        patterns: Parsed patterns.

    Returns:
        One row per matched row, with `pattern` and `position` (the index of the
        transaction line in the pattern) and `row_id`. Identical raw rows all
        match, and sorting keeps the result order stable.
    """
    lookup_rows = [
        (pattern.index, position, *fields)
        for pattern in patterns
        for position, fields in enumerate(pattern.transactions)
    ]
    schema = {
        "pattern": pl.Int64,
        "position": pl.Int64,
        **dict.fromkeys(TRANS_COLUMNS, pl.String),
    }
    lookup = pl.DataFrame(lookup_rows, schema=schema, orient="row")
    return (
        _scan_raw(trans_path, TRANS_COLUMNS)
        .join(lookup.lazy(), on=list(TRANS_COLUMNS), how="inner")
        .select("pattern", "position", "row_id")
        .sort("pattern", "position", "row_id")
        .collect()
    )


def _summarise_patterns(
    patterns: Sequence[Pattern], matches: pl.DataFrame, light: pl.DataFrame
) -> tuple[list[_PatternSummary], list[int], list[int]]:
    """Count matched illicit rows per day block for each complete pattern.

    Returns:
        The summaries of patterns whose every line matched, then the indexes of
        patterns with no matched line, then the indexes of partly matched ones.
    """
    lines_found = dict(
        matches.group_by("pattern").agg(pl.col("position").n_unique()).iter_rows()
    )
    block_counts: dict[int, Counter[str]] = defaultdict(Counter)
    matched_blocks = matches.join(light.select("row_id", "block"), on="row_id")
    for pattern, block in matched_blocks.select("pattern", "block").iter_rows():
        block_counts[pattern][block] += 1

    summaries: list[_PatternSummary] = []
    unmatched: list[int] = []
    partial: list[int] = []
    for pattern in patterns:
        found = lines_found.get(pattern.index, 0)
        if found == 0:
            unmatched.append(pattern.index)
        elif found < len(pattern.transactions):
            partial.append(pattern.index)
        else:
            summaries.append(
                _PatternSummary(
                    index=pattern.index,
                    typology=pattern.typology,
                    length=len(pattern.transactions),
                    block_counts=block_counts[pattern.index],
                )
            )
    return summaries, unmatched, partial


def _choose_patterns(summaries: Sequence[_PatternSummary]) -> list[_PatternSummary]:
    """Pick patterns with the fewest transactions until the blocks are covered.

    Two patterns per typology come first, where the typology has that many.
    Then the smallest pattern that adds illicit rows to a short block is taken,
    until every block meets its minimum.

    Returns:
        The chosen patterns in file order.
    """
    ordered = sorted(summaries, key=lambda s: (s.length, s.index))
    chosen: list[_PatternSummary] = []
    for typology in TYPOLOGIES:
        of_type = [s for s in ordered if s.typology == typology]
        chosen.extend(of_type[:PATTERNS_PER_TYPOLOGY])
    chosen_indices = {s.index for s in chosen}
    while True:
        totals: Counter[str] = Counter()
        for summary in chosen:
            totals.update(summary.block_counts)
        short = [label for label in BLOCK_LABELS if totals[label] < _minimum(label)]
        if not short:
            break
        picked = next(
            (
                s
                for s in ordered
                if s.index not in chosen_indices
                and any(s.block_counts[label] for label in short)
            ),
            None,
        )
        if picked is None:
            msg = f"no pattern adds illicit rows to the short blocks {short}"
            raise RuntimeError(msg)
        chosen.append(picked)
        chosen_indices.add(picked.index)
    return sorted(chosen, key=lambda s: s.index)


def _minimum(label: str) -> int:
    """Return the illicit row minimum of one day block or the Tail."""
    return MIN_ILLICIT_TAIL if label == "tail" else MIN_ILLICIT_PER_BLOCK


def _accounts_of(light: pl.DataFrame, row_ids: Sequence[int]) -> list[str]:
    """Return the sorted account keys that appear in the given rows."""
    rows = light.filter(pl.col("row_id").is_in(list(row_ids)))
    return sorted(set(rows["src"].to_list()) | set(rows["dst"].to_list()))


def _account_streams(light: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the rows into the send stream and the receive stream of each Account.

    Self-transfers are left out, so every stream pair involves two Accounts.

    Returns:
        Two frames with `account`, `peer`, `minute` and `row_id`: sends, then
        receives.
    """
    transfers = light.filter(pl.col("src") != pl.col("dst"))
    sends = transfers.select(
        account=pl.col("src"),
        peer=pl.col("dst"),
        minute=pl.col("minute"),
        row_id=pl.col("row_id"),
    )
    receives = transfers.select(
        account=pl.col("dst"),
        peer=pl.col("src"),
        minute=pl.col("minute"),
        row_id=pl.col("row_id"),
    )
    return sends, receives


def _gap_candidates(stream: pl.DataFrame, gap: int) -> list[_Candidate]:
    """Return pairs of events on one Account stream exactly `gap` minutes apart."""
    early = stream.with_columns(target=pl.col("minute") + gap)
    late = stream.select(
        pl.col("account"),
        target=pl.col("minute"),
        late_peer=pl.col("peer"),
        late_row_id=pl.col("row_id"),
    )
    pairs = early.join(late, on=["account", "target"], how="inner")
    return [
        _Candidate(
            rows=(row["row_id"], row["late_row_id"]),
            involved=tuple(sorted({row["account"], row["peer"], row["late_peer"]})),
            account=row["account"],
            minute=row["target"],
        )
        for row in pairs.iter_rows(named=True)
    ]


def _triple_candidates(sends: pl.DataFrame) -> list[_Candidate]:
    """Return Accounts with sends at 0, 60, 1,440 and 10,080 minutes apart."""
    grouped = (
        sends.group_by("account")
        .agg(pl.col("minute"), pl.col("row_id"), pl.col("peer"))
        .filter(pl.col("minute").list.len() > len(GAP_MINUTES))
    )
    candidates: list[_Candidate] = []
    for account, minutes, row_ids, peers in grouped.iter_rows():
        first: dict[int, tuple[int, str]] = {}
        for minute, row_id, peer in sorted(zip(minutes, row_ids, peers, strict=True)):
            first.setdefault(minute, (row_id, peer))
        for base, (base_row, base_peer) in first.items():
            if not all(base + gap in first for gap in GAP_MINUTES):
                continue
            later = [first[base + gap] for gap in GAP_MINUTES]
            candidates.append(
                _Candidate(
                    rows=(base_row, *(row for row, _ in later)),
                    involved=tuple(
                        sorted({account, base_peer, *(peer for _, peer in later)})
                    ),
                    account=account,
                    minute=base + GAP_MINUTES[-1],
                )
            )
    return candidates


def _same_minute_candidates(sends: pl.DataFrame) -> list[_Candidate]:
    """Return pairs of sends from one Account in one minute to different peers."""
    pairs = sends.join(sends, on=["account", "minute"], suffix="_other").filter(
        pl.col("row_id") < pl.col("row_id_other"),
        pl.col("peer") != pl.col("peer_other"),
    )
    return [
        _Candidate(
            rows=(row["row_id"], row["row_id_other"]),
            involved=tuple(sorted({row["account"], row["peer"], row["peer_other"]})),
        )
        for row in pairs.iter_rows(named=True)
    ]


def _two_bank_candidates(light: pl.DataFrame) -> list[_Candidate]:
    """Return account numbers that appear under two or more banks."""
    accounts = pl.concat(
        [
            light.select(bank=pl.col("from_bank"), number=pl.col("from_account")),
            light.select(bank=pl.col("to_bank"), number=pl.col("to_account")),
        ]
    ).unique()
    shared = (
        accounts.group_by("number")
        .agg(pl.col("bank").unique())
        .filter(pl.col("bank").list.len() > 1)
    )
    return [
        _Candidate(
            rows=(),
            involved=tuple(sorted(f"{bank}:{row['number']}" for bank in row["bank"])),
        )
        for row in shared.iter_rows(named=True)
    ]


def _pick_candidate(
    candidates: Sequence[_Candidate],
    *,
    degree: dict[str, int],
    neighbour_events: pl.DataFrame | None,
) -> _Candidate:
    """Return the cheapest candidate, optionally with a nearby Transaction.

    Cost is the sum of the involved Accounts' degrees, so the neighbourhood
    that gets kept stays small. When `neighbour_events` is given, the candidate
    must have another Transaction of its Account within the neighbour window of
    its later Transaction.
    """
    ranked = sorted(
        candidates,
        key=lambda c: (sum(degree[key] for key in c.involved), c.rows),
    )
    _require(bool(ranked), "no candidate exists for an edge case in the data")
    if neighbour_events is None:
        return ranked[0]
    pool = ranked[:PICK_POOL_SIZE]
    accounts = sorted({c.account for c in pool if c.account is not None})
    events: dict[str, list[tuple[int, int]]] = defaultdict(list)
    near = neighbour_events.filter(pl.col("account").is_in(accounts))
    for account, minute, row_id in near.iter_rows():
        events[account].append((minute, row_id))
    for candidate in pool:
        if _has_neighbour(candidate, events):
            return candidate
    msg = "no candidate has another Transaction nearby"
    raise RuntimeError(msg)


def _has_neighbour(
    candidate: _Candidate, events: dict[str, list[tuple[int, int]]]
) -> bool:
    """Return true when the candidate's Account has an event nearby."""
    if candidate.account is None or candidate.minute is None:
        return False
    return any(
        abs(minute - candidate.minute) <= NEIGHBOUR_WINDOW_MINUTES
        and row_id not in candidate.rows
        for minute, row_id in events.get(candidate.account, [])
    )


def _edge_case_rows(light: pl.DataFrame) -> set[int]:
    """Return the rows of the edge cases the tests need, with neighbours.

    Each edge case keeps every Transaction of the Accounts involved in it.
    """
    sends, receives = _account_streams(light)
    both = pl.concat([sends, receives]).select("account", "minute", "row_id")
    degree = dict(both.group_by("account").agg(pl.len().alias("degree")).iter_rows())

    chosen: list[_Candidate] = [
        _pick_candidate(_triple_candidates(sends), degree=degree, neighbour_events=both)
    ]
    chosen.append(
        _pick_candidate(
            _gap_candidates(receives, GAP_MINUTES[0]),
            degree=degree,
            neighbour_events=both,
        )
    )
    chosen.append(
        _pick_candidate(
            _same_minute_candidates(sends), degree=degree, neighbour_events=None
        )
    )
    chosen.append(
        _pick_candidate(
            _two_bank_candidates(light), degree=degree, neighbour_events=None
        )
    )

    rows: set[int] = set()
    for candidate in chosen:
        logger.info("Edge case: accounts %s", list(candidate.involved))
        rows.update(_neighbourhood(light, candidate.involved))
    return rows


def _neighbourhood(light: pl.DataFrame, accounts: Sequence[str]) -> list[int]:
    """Return the row ids of every Transaction that touches one of `accounts`."""
    touched = light.filter(
        pl.col("src").is_in(list(accounts)) | pl.col("dst").is_in(list(accounts))
    )
    return touched["row_id"].to_list()


def _tail_rows(light: pl.DataFrame, rng: np.random.Generator) -> list[int]:
    """Draw non-illicit Tail rows at random, with the seeded generator."""
    tail = light.filter((pl.col("split") == "tail") & ~pl.col("illicit"))
    candidates = np.sort(tail["row_id"].to_numpy())
    picked = rng.choice(candidates, size=TAIL_ROWS, replace=False)
    return sorted(int(row_id) for row_id in picked)


def _background_rows(
    light: pl.DataFrame,
    taken: set[int],
    needed: int,
    rng: np.random.Generator,
) -> list[int]:
    """Draw random rows from days 1 to 10 in proportion to each day's size.

    Every day gets at least one row. The draw is seeded, and each day samples
    from its row ids in sorted order.
    """
    pool = light.filter(
        (pl.col("split") != "tail") & ~pl.col("row_id").is_in(sorted(taken))
    )
    total = pool.height
    drawn: list[int] = []
    for day in sorted(pool["day"].unique().to_list()):
        ids = np.sort(pool.filter(pl.col("day") == day)["row_id"].to_numpy())
        count = max(1, round(needed * len(ids) / total))
        picked = rng.choice(ids, size=count, replace=False)
        drawn.extend(int(row_id) for row_id in picked)
    return drawn


def _bank_key(bank: pl.Expr) -> pl.Expr:
    """Return a bank id as its number without leading zeros.

    The transactions file zero-pads some bank ids, while the accounts file
    writes the same banks without padding. Matching uses this key on both sides.
    """
    return bank.cast(pl.Int64).cast(pl.String)


def _log_padded_bank_ids(light: pl.DataFrame) -> None:
    """Log the transaction bank ids that differ only by leading zeros."""
    raw = set(light["from_bank"].unique().to_list()) | set(
        light["to_bank"].unique().to_list()
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for bank in sorted(raw):
        groups[str(int(bank))].append(bank)
    shared = {key: texts for key, texts in groups.items() if len(texts) > 1}
    logger.info("Bank ids that share a key without leading zeros: %s", shared)


def _account_line_ids(accounts_path: Path, kept: pl.DataFrame) -> list[int]:
    """Return the accounts file lines of every Account in the kept rows.

    Raises:
        RuntimeError: An Account in the kept rows has no line in the accounts file.
    """
    pairs = (
        pl.concat(
            [
                kept.select(
                    bank_id=_bank_key(pl.col("from_bank")),
                    account_number=pl.col("from_account"),
                ),
                kept.select(
                    bank_id=_bank_key(pl.col("to_bank")),
                    account_number=pl.col("to_account"),
                ),
            ]
        )
        .unique()
        .sort("bank_id", "account_number")
    )
    accounts = _scan_raw(accounts_path, ACCOUNT_COLUMNS).select(
        "row_id",
        raw_bank_id=pl.col("bank_id"),
        bank_id=_bank_key(pl.col("bank_id")),
        account_number=pl.col("account_number"),
    )
    matched = accounts.join(
        pairs.lazy(), on=["bank_id", "account_number"], how="inner"
    ).collect()
    found = matched.select("bank_id", "account_number").unique().height
    _require(
        found == pairs.height,
        f"{pairs.height - found} Accounts in the sample have no accounts line",
    )
    raw_per_key = matched.select("bank_id", "account_number", "raw_bank_id").unique()
    _require(
        raw_per_key.height == found,
        "the accounts file writes one bank id under two texts",
    )
    return matched["row_id"].unique().sort().to_list()


def _copy_selected_lines(source: Path, destination: Path, row_ids: set[int]) -> int:
    """Copy the header and the selected data lines of a CSV as raw bytes.

    Args:
        source: Headed file with one record per line.
        destination: Output file, replaced if it exists.
        row_ids: Zero-based data row indexes to keep.

    Returns:
        The number of data lines in `source`.

    Raises:
        RuntimeError: Not every selected row was found in `source`.
    """
    data_lines = 0
    written = 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        writer.write(_with_newline(reader.readline()))
        for index, line in enumerate(reader):
            data_lines += 1
            if index in row_ids:
                writer.write(_with_newline(line))
                written += 1
    _require(
        written == len(row_ids),
        f"wrote {written} of {len(row_ids)} selected rows to {destination.name}",
    )
    return data_lines


def _verify_sample(
    kept: pl.DataFrame,
    *,
    matches: pl.DataFrame,
    chosen_indices: set[int],
    pattern_rows: Sequence[int],
    closure: Sequence[int],
) -> None:
    """Assert the properties the tests and the README rely on.

    Raises:
        RuntimeError: One property is missing from the kept rows.
    """
    kept_ids = set(kept["row_id"].to_list())
    _require(
        set(pattern_rows) <= kept_ids and set(closure) <= kept_ids,
        "pattern closure is missing rows",
    )
    chosen_rows = set(
        matches.filter(pl.col("pattern").is_in(sorted(chosen_indices)))[
            "row_id"
        ].to_list()
    )
    _require(chosen_rows <= kept_ids, "a chosen pattern transaction is missing")

    illicit = kept.filter(pl.col("illicit"))
    counts = Counter(illicit["block"].to_list())
    for label in BLOCK_LABELS:
        _require(
            counts[label] >= _minimum(label),
            f"block {label} has {counts[label]} illicit rows, below {_minimum(label)}",
        )
    days = set(kept["day"].unique().to_list())
    _require(
        set(range(1, DAY_BLOCKS[-1][1] + 1)) <= days,
        "some day in 1 to 10 has no rows",
    )

    sends, receives = _account_streams(kept)
    _require(
        bool(_triple_candidates(sends)),
        "no Account has sends 60, 1,440 and 10,080 minutes after one send",
    )
    _require(
        bool(_gap_candidates(receives, GAP_MINUTES[0])),
        "no receive pair 60 minutes apart",
    )
    _require(
        bool(_same_minute_candidates(sends)),
        "no Account has two sends in one minute",
    )
    _require(
        bool(_two_bank_candidates(kept)),
        "no account number appears under two banks",
    )


def _readme(
    *,
    kept_light: pl.DataFrame,
    seed: int,
    rows: int,
    pattern_count: int,
) -> str:
    """Return the README text for the sample directory."""
    stats = {
        row["block"]: (row["rows"], row["illicit"])
        for row in kept_light.group_by("block")
        .agg(pl.len().alias("rows"), pl.col("illicit").sum().alias("illicit"))
        .iter_rows(named=True)
    }
    table = "\n".join(
        f"| {label} | {stats.get(label, (0, 0))[0]} | {stats.get(label, (0, 0))[1]} |"
        for label in BLOCK_LABELS
    )
    total_rows = kept_light.height
    total_illicit = int(kept_light["illicit"].sum())
    return f"""# AMLworld HI-Small sample

This directory holds a sample of the IBM AMLworld HI-Small Variant. The raw
data comes from the Kaggle dataset `{KAGGLE_DATASET}`.

The rows are real and unmodified. The files keep the names, header lines and
row text of the full data, so the same loader reads the sample and the full
data.

## Files

| File | Contents |
| --- | --- |
| `{TRANS_FILE}` | {total_rows} transactions in file order, {total_illicit} illicit |
| `{ACCOUNTS_FILE}` | The accounts that appear in the transactions |
| `{PATTERNS_FILE}` | {pattern_count} laundering attempts from the Patterns file |
| `LICENSE` | The CDLA-Sharing-1.0 license text for the AMLworld data |

## Selection rule

The builder reads the full data and applies these steps in order.

1. A Pattern transaction line matches a raw row when all eleven columns are
   equal. A Pattern counts only when every one of its lines matches.
2. The two Patterns with the fewest transactions are taken for each of the
   eight typologies. More Patterns are added, smallest first, until each day
   block has at least {MIN_ILLICIT_PER_BLOCK} illicit transactions and the Tail
   has at least {MIN_ILLICIT_TAIL}.
3. Every transaction between two Accounts of the chosen Patterns is kept.
4. Edge-case rows are added with every transaction of the Accounts involved:
   a send pair and a receive pair one hour apart, send pairs 1,440 and 10,080
   minutes apart on the same Account, two sends in one minute from one
   Account, and every transaction of an account number used under two banks.
5. {TAIL_ROWS} non-illicit Tail transactions are drawn at random.
6. Other transactions from days 1 to 10 are drawn at random, in proportion to
   the size of each day, until the file reaches {rows} rows.

The Account identity is the pair of bank and account number everywhere. The
accounts file writes bank ids without leading zeros, so the builder matches
its lines by the numeric bank id. The transactions file keeps the zeros.

## Counts per day block

Days 1 to 10 form the blocks below. The Tail is every row after day 10.

| Day block | Rows | Illicit rows |
| --- | --- | --- |
{table}

## Regenerating the sample

Run `make sample` from the repository root. The command reads the three raw
files from `data/` and rewrites this directory. It uses the seed {seed}. The
same raw files, seed and row target give byte-identical files.
"""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sample builder from the command line.

    Args:
        argv: Arguments without the program name. Defaults to `sys.argv`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    build_sample(
        data_dir=args.data_dir,
        out_dir=args.out,
        seed=args.seed,
        rows=args.rows,
    )


if __name__ == "__main__":
    main()
