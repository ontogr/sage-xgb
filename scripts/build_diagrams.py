"""Build the README diagrams as static light SVG files.

Each diagram is described once as data: boxes, chips, edges and text. The
builder draws it with one light palette and writes `<name>.svg` into
`docs/img/readme/`. The files use
system fonts, no scripts and no animation, so they render the same inside an
`<img>` tag on any host.

Run `uv run python scripts/build_diagrams.py` after the code or the diagrams
change.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape


logger = logging.getLogger(__name__)

SANS = (
    'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, '
    "Arial, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

PALETTE: dict[str, str] = {
    "panel": "#faf8f3",
    "panel_line": "#e4e0d6",
    "box": "#f3f0e9",
    "box_line": "#cbc6b9",
    "chip": "#ffffff",
    "ink": "#1f2937",
    "muted": "#6b655f",
    "blue": "#1d4ed8",
    "tint": "#eaf0fe",
    "on_blue": "#ffffff",
}

CSS = """
.panel{{fill:{panel};stroke:{panel_line}}}
.title{{font:600 16px {sans};fill:{ink}}}
.stage{{font:600 11px {mono};letter-spacing:.08em;fill:{muted}}}
.box{{fill:{box};stroke:{box_line}}}
.box-title{{font:600 13px {sans};fill:{ink}}}
.box-sub{{font:10px {mono};fill:{muted}}}
.chip{{fill:{chip};stroke:{box_line}}}
.chip-name{{font:600 12px {sans};fill:{ink}}}
.chip-tag{{font:10px {mono};fill:{blue}}}
.node{{fill:{tint};stroke:{blue};stroke-width:1.25}}
.node-name{{font:600 13px {sans};fill:{ink}}}
.node-sub{{font:10px {mono};fill:{muted}}}
.focal{{fill:{blue};stroke:{blue};stroke-width:1.25}}
.focal-name{{font:600 13px {sans};fill:{on_blue}}}
.focal-sub{{font:10px {mono};fill:{on_blue};opacity:.9}}
.band{{fill:none;stroke:{box_line};stroke-dasharray:3 4}}
.edge{{fill:none;stroke:{muted};stroke-width:1.5}}
.edge-blue{{fill:none;stroke:{blue};stroke-width:1.5}}
.edge-dash{{fill:none;stroke:{blue};stroke-width:1.5;stroke-dasharray:5 4}}
.edge-soft{{fill:none;stroke:{muted};stroke-width:1.25;stroke-dasharray:5 4}}
.tick{{stroke:{muted};stroke-width:1.25}}
.elabel{{font:10px {mono};fill:{muted};letter-spacing:.04em}}
.elabel-blue{{font:10px {mono};fill:{blue};letter-spacing:.04em}}
.note{{font:12px {sans};fill:{muted}}}
.note-ink{{font:600 12px {sans};fill:{ink}}}
.arrowhead{{fill:{muted}}}
.arrowhead-blue{{fill:{blue}}}
.dot{{fill:{chip};stroke:{muted};stroke-width:1.5}}
.dot-focal{{fill:{blue};stroke:{blue};stroke-width:1.5}}
.dot-open{{fill:{panel};stroke:{blue};stroke-width:1.75}}
.dot-num{{font:600 10px {mono};fill:{ink}}}
.dot-num-focal{{font:600 10px {mono};fill:{on_blue}}}
"""

BOX_CLASSES = {
    "box": ("box", "box-title", "box-sub"),
    "node": ("node", "node-name", "node-sub"),
    "focal": ("focal", "focal-name", "focal-sub"),
}

# Rough glyph widths in pixels per character, to catch text that overflows.
WIDTH_PER_CHAR = {"title": 8.6, "name": 7.8, "mono": 6.3, "note": 6.6, "label": 6.6}

Point = tuple[float, float]
Area = tuple[float, float, float, float]


@dataclass
class Svg:
    """One diagram under construction."""

    slug: str
    title: str
    description: str
    width: int
    height: int
    parts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, text: str, kind: str, limit: float) -> None:
        """Record a warning when `text` is probably wider than `limit`."""
        estimate = len(text) * WIDTH_PER_CHAR[kind]
        if estimate > limit:
            self.warnings.append(
                f"{self.slug}: {text!r} ~{estimate:.0f}px > {limit:.0f}"
            )

    def text(
        self, x: float, y: float, text: str, css: str, anchor: str = "start"
    ) -> None:
        """Add a text element."""
        extra = "" if anchor == "start" else f' text-anchor="{anchor}"'
        self.parts.append(
            f'<text class="{css}" x="{x:g}" y="{y:g}"{extra}>{escape(text)}</text>'
        )

    def header(self, title: str, subtitle: str) -> None:
        """Add the diagram title and its one-line description."""
        self.text(28, 36, title, "title")
        self.text(28, 58, subtitle, "note")

    def box(
        self,
        area: Area,
        title: str,
        subs: Sequence[str] = (),
        kind: str = "box",
    ) -> None:
        """Add a titled box with mono sublabel lines."""
        x, y, w, h = area
        shape, name, sub = BOX_CLASSES[kind]
        self.check(title, "name", w - 28)
        self.rect(area, shape, 6)
        self.text(x + 14, y + 24, title, name)
        for index, line in enumerate(subs):
            self.check(line, "mono", w - 28)
            self.text(x + 14, y + 44 + 14 * index, line, sub)
        needed = 44 + 14 * max(len(subs) - 1, 0) + 14
        if subs and needed > h:
            self.warnings.append(f"{self.slug}: box {title!r} needs height {needed}")

    def chip(self, area: Area, name: str, tag: str = "") -> None:
        """Add a white chip with a name and an optional blue tag on the right."""
        x, y, w, h = area
        self.rect(area, "chip", 5)
        self.check(name + tag, "mono", w - 24)
        self.text(x + 12, y + h / 2 + 4, name, "chip-name")
        if tag:
            self.text(x + w - 12, y + h / 2 + 4, tag, "chip-tag", "end")

    def rect(self, area: Area, css: str, rx: int = 6) -> None:
        """Add a plain rectangle with a style class."""
        x, y, w, h = area
        box = f'x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}"'
        self.parts.append(f'<rect class="{css}" {box} rx="{rx}"/>')

    def edge(
        self, points: Sequence[Point], css: str = "edge", arrow: bool = True
    ) -> None:
        """Add an orthogonal connector with rounded corners and an arrowhead."""
        path = _rounded_path(points, 8)
        head = "-blue" if css in ("edge-blue", "edge-dash") else ""
        marker = f' marker-end="url(#ah{head}-{self.slug})"' if arrow else ""
        self.parts.append(f'<path class="{css}" d="{path}"{marker}/>')

    def line(
        self, x1: float, y1: float, x2: float, y2: float, css: str = "tick"
    ) -> None:
        """Add a straight line segment."""
        self.parts.append(
            f'<line class="{css}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"/>'
        )

    def dot(
        self, x: float, y: float, css: str, label: str = "", label_css: str = "dot-num"
    ) -> None:
        """Add a circle, with an optional centered label."""
        self.parts.append(f'<circle class="{css}" cx="{x:g}" cy="{y:g}" r="9"/>')
        if label:
            self.text(x, y + 3.5, label, label_css, "middle")

    def render(self) -> str:
        """Return the SVG document."""
        css = CSS.format(sans=SANS, mono=MONO, **PALETTE)
        ident = self.slug
        markers = "".join(
            f'<marker id="ah{tone}-{self.slug}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto">'
            f'<path class="arrowhead{tone}" d="M0 0L10 5L0 10z"/></marker>'
            for tone in ("", "-blue")
        )
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" role="img" '
            f'aria-labelledby="{ident}-title {ident}-desc">\n'
            f'<title id="{ident}-title">{escape(self.title)}</title>\n'
            f'<desc id="{ident}-desc">{escape(self.description)}</desc>\n'
            f"<style>{css}</style>\n<defs>{markers}</defs>\n"
            f'<rect class="panel" x="0.5" y="0.5" width="{self.width - 1}" '
            f'height="{self.height - 1}" rx="10"/>\n{body}\n</svg>\n'
        )


def _rounded_path(points: Sequence[Point], radius: float) -> str:
    """Return an SVG path through `points` with rounded right-angle corners."""
    commands = [f"M{points[0][0]:g} {points[0][1]:g}"]
    for index in range(1, len(points) - 1):
        (px, py), (cx, cy), (nx, ny) = (
            points[index - 1],
            points[index],
            points[index + 1],
        )
        before = min(radius, abs(cx - px) + abs(cy - py)) / max(
            abs(cx - px) + abs(cy - py), 1
        )
        after = min(radius, abs(nx - cx) + abs(ny - cy)) / max(
            abs(nx - cx) + abs(ny - cy), 1
        )
        start = (cx - (cx - px) * before, cy - (cy - py) * before)
        end = (cx + (nx - cx) * after, cy + (ny - cy) * after)
        commands.append(
            f"L{start[0]:g} {start[1]:g}Q{cx:g} {cy:g} {end[0]:g} {end[1]:g}"
        )
    commands.append(f"L{points[-1][0]:g} {points[-1][1]:g}")
    return "".join(commands)


def pipeline() -> Svg:
    """Draw the four `run` stages, their files, and the separate commands."""
    svg = Svg(
        "pipeline",
        "The sage-xgb pipeline",
        "prepare feeds features and gnn, and both feed train-xgb. train-xgb writes the "
        "model bundle that report reads. tune and report are separate commands.",
        880,
        530,
    )
    svg.header(
        "The sage-xgb pipeline",
        "Four stages run in order. Each writes parquet or JSON files and a manifest.",
    )
    svg.rect((180, 78, 684, 276), "band", 10)
    svg.text(196, 100, "SAGE-XGB RUN", "stage")
    svg.chip((28, 188, 132, 34), "Trans.csv", "raw")
    svg.text(28, 240, "rows are not in time order", "box-sub")
    svg.box(
        (196, 150, 190, 100),
        "prepare",
        [
            "sort by minute and row id",
            "day, split, USD, node ids",
            "transactions.parquet",
            "accounts.parquet",
        ],
    )
    svg.box(
        (428, 108, 190, 86),
        "features",
        ["windows over [t - w, t)", "49 columns, no labels", "features.parquet"],
    )
    svg.box(
        (428, 214, 190, 100),
        "gnn (GraphSAGE)",
        [
            "3 forward-chained fits",
            "score + embedding per row",
            "gnn_scores.parquet",
            "gnn_fit_1 to 3.pt",
        ],
        "node",
    )
    svg.box(
        (676, 150, 176, 114),
        "train-xgb",
        [
            "early stop: validation",
            "threshold: val F1",
            "booster.json",
            "feature_columns.json",
            "metadata.json",
        ],
        "focal",
    )
    svg.edge([(160, 205), (196, 205)])
    svg.edge([(386, 178), (404, 178), (404, 151), (428, 151)])
    svg.edge([(386, 222), (404, 222), (404, 264), (428, 264)])
    svg.edge([(618, 151), (640, 151), (640, 190), (676, 190)])
    svg.edge([(618, 264), (654, 264), (654, 226), (676, 226)], "edge-blue")
    svg.text(624, 284, "OUT-OF-FOLD SCORES", "elabel-blue")
    svg.edge([(284, 250), (284, 334), (764, 334), (764, 264)], "edge-soft")
    svg.text(524, 328, "TRAIN AND VALIDATION LABELS", "elabel", "middle")
    svg.text(28, 396, "SEPARATE COMMANDS, NOT PART OF RUN", "stage")
    svg.box(
        (196, 408, 300, 72),
        "tune",
        ["Optuna on validation PR-AUC", "writes best.yaml, a full config"],
    )
    svg.box(
        (644, 408, 208, 100),
        "report",
        [
            "test and Tail, scored once",
            "reads features, GNN scores",
            "report.json",
            "history.jsonl",
        ],
    )
    svg.edge([(816, 264), (816, 408)])
    svg.text(806, 384, "BUNDLE", "elabel", "end")
    return svg


def split() -> Svg:
    """Draw the day axis with the train blocks, validation, test and Tail."""
    svg = Svg(
        "split",
        "Day-aligned split",
        "Days 1 to 6 are train, in three cross-fit blocks. Days 7 and 8 are "
        "validation, "
        "days 9 and 10 are test, and days 11 to 18 are the Tail, scored apart.",
        880,
        340,
    )
    svg.header(
        "Split by whole days",
        "HI-Small spans 2022-09-01 to 2022-09-18. Every cut falls on a day boundary.",
    )
    left, day = 44, 44
    for name, first, last in (
        ("TRAIN", 1, 6),
        ("VALIDATION", 7, 8),
        ("TEST", 9, 10),
        ("TAIL", 11, 18),
    ):
        x1, x2 = left + (first - 1) * day + 2, left + last * day - 2
        svg.line(x1, 106, x2, 106)
        svg.line(x1, 106, x1, 112)
        svg.line(x2, 106, x2, 112)
        svg.text((x1 + x2) / 2, 96, name, "stage", "middle")
    cells = [
        (1, 2, "Block 1", "node"),
        (3, 4, "Block 2", "node"),
        (5, 6, "Block 3", "node"),
        (7, 8, "Val", "box"),
        (9, 10, "Test", "focal"),
        (11, 18, "Tail, scored apart", "box"),
    ]
    for first, last, label, kind in cells:
        x = left + (first - 1) * day
        width = (last - first + 1) * day
        shape, name, _ = BOX_CLASSES[kind]
        svg.rect((x + 2, 120, width - 4, 44), shape, 6)
        svg.text(x + width / 2, 147, label, name, "middle")
    for number in range(1, 19):
        svg.text(left + (number - 0.5) * day, 184, str(number), "box-sub", "middle")
    svg.text(28, 206, "day number since 2022-09-01", "box-sub")
    svg.box(
        (28, 226, 200, 96),
        "Train",
        ["days 1-6, three blocks", "fits GraphSAGE GNNs", "trains XGBoost on days 3-6"],
        "node",
    )
    svg.box(
        (240, 226, 200, 96),
        "Validation",
        ["days 7-8", "early stopping", "decision threshold"],
    )
    svg.box(
        (452, 226, 200, 96),
        "Test",
        ["days 9-10", "opened once, by report", "never used to tune"],
        "focal",
    )
    svg.box(
        (664, 226, 188, 96),
        "Tail",
        [
            "days 11-18, 1,108 rows",
            "most rows are laundering",
            "reported as its own slice",
        ],
    )
    return svg


def crossfit() -> Svg:
    """Draw which blocks each GNN fit trains on and scores."""
    svg = Svg(
        "crossfit",
        "Forward-chained cross-fit",
        "Fit 1 trains on block 1 and scores block 2. Fit 2 trains on blocks 1 and "
        "2 and "
        "scores block 3. Fit 3 trains on blocks 1 to 3 and scores validation, test and "
        "Tail. XGBoost trains on the out-of-fold scores of days 3 to 6.",
        880,
        390,
    )
    svg.header(
        "Cross-fit over day blocks",
        "Each GraphSAGE GNN scores only days that come after every day it trained on.",
    )
    columns = [
        ("BLOCK 1", "days 1-2"),
        ("BLOCK 2", "days 3-4"),
        ("BLOCK 3", "days 5-6"),
        ("VALIDATION", "days 7-8"),
        ("TEST", "days 9-10"),
        ("TAIL", "days 11-18"),
    ]
    x0, step, width = 150, 116, 108
    for index, (name, days) in enumerate(columns):
        x = x0 + index * step + width / 2
        svg.text(x, 92, name, "stage", "middle")
        svg.text(x, 108, days, "box-sub", "middle")
    plan = [
        ("Fit 1", "block 1", {0}, {1}),
        ("Fit 2", "blocks 1-2", {0, 1}, {2}),
        ("Fit 3", "blocks 1-3", {0, 1, 2}, {3, 4, 5}),
    ]
    for row, (name, sub, trains, scores) in enumerate(plan):
        y = 124 + row * 52
        svg.text(28, y + 20, name, "note-ink")
        svg.text(28, y + 34, sub, "box-sub")
        for index in sorted(trains | scores):
            kind = "node" if index in trains else "box"
            shape, label_css, _ = BOX_CLASSES[kind]
            svg.rect((x0 + index * step, y, width, 42), shape, 6)
            svg.text(
                x0 + index * step + width / 2,
                y + 26,
                "trains" if index in trains else "scores",
                label_css,
                "middle",
            )
    svg.text(28, 300, "WHAT XGBOOST USES", "stage")
    svg.box(
        (x0 + step, 312, 2 * step - 8, 60),
        "Training rows",
        ["out-of-fold scores, days 3-6"],
        "focal",
    )
    svg.box((x0 + 3 * step, 312, width, 60), "Early stop", ["threshold"])
    svg.box(
        (x0 + 4 * step, 312, 2 * step - 8, 60), "Report only", ["test and Tail scores"]
    )
    return svg


def windows() -> Svg:
    """Draw the half-open trailing window on a worked four-send example."""
    svg = Svg(
        "windows",
        "Trailing window rule",
        "For a send at time t, the window covers t minus w up to but not including t. "
        "A send exactly w earlier counts, and sends in the same minute never "
        "count each other.",
        880,
        470,
    )
    svg.header(
        "Trailing window [t - w, t)",
        "One sender, four sends. The count is of the sender's earlier sends in "
        "the last hour.",
    )
    svg.dot(548, 50, "dot-open")
    svg.text(564, 54, "row at t: excluded", "note")
    svg.dot(704, 50, "dot-focal")
    svg.text(720, 54, "row at t - w: counted", "note")
    axis = 230

    def x_of(minutes: int) -> int:
        return 100 + minutes * 5

    svg.line(80, axis, 790, axis, "edge")
    for minutes, label in (
        (0, "09:00"),
        (30, "09:30"),
        (60, "10:00"),
        (90, "10:30"),
        (120, "11:00"),
    ):
        svg.line(x_of(minutes), axis - 6, x_of(minutes), axis + 6)
        svg.text(x_of(minutes), axis + 26, label, "elabel", "middle")
    svg.edge([(x_of(60), 130), (x_of(120), 130)], "edge-blue", arrow=False)
    svg.edge([(x_of(60), 130), (x_of(60), axis - 44)], "edge-dash", arrow=False)
    svg.edge([(x_of(120), 130), (x_of(120), axis - 44)], "edge-dash", arrow=False)
    svg.dot(x_of(60), 130, "dot-focal")
    svg.dot(x_of(120), 130, "dot-open")
    svg.text(x_of(90), 116, "WINDOW OF SEND 4: [10:00, 11:00)", "elabel-blue", "middle")
    svg.text(x_of(90), 154, "contains sends 1, 2, 3: count 3", "elabel-blue", "middle")
    svg.dot(x_of(60), axis - 28, "dot", "1")
    svg.dot(x_of(90), axis - 28, "dot", "2")
    svg.dot(x_of(90), axis - 52, "dot", "3")
    svg.dot(x_of(120), axis - 28, "dot-focal", "4", "dot-num-focal")
    svg.text(x_of(90) - 16, axis - 50, "same minute", "box-sub", "end")
    low = 300
    svg.edge([(x_of(30), low), (x_of(90), low)], "edge-soft", arrow=False)
    svg.edge([(x_of(30), low), (x_of(30), axis + 40)], "edge-soft", arrow=False)
    svg.edge([(x_of(90), low), (x_of(90), axis + 40)], "edge-soft", arrow=False)
    svg.text(
        x_of(60),
        low + 22,
        "WINDOW OF SENDS 2 AND 3: [09:30, 10:30)",
        "elabel",
        "middle",
    )
    svg.text(
        x_of(60), low + 38, "contains send 1 only: count 1 each", "elabel", "middle"
    )
    results = [
        ("Send 1, 10:00", "count 0", "chip"),
        ("Send 2, 10:30", "count 1", "chip"),
        ("Send 3, 10:30", "count 1", "chip"),
        ("Send 4, 11:00", "count 3", "node"),
    ]
    for index, (name, tag, kind) in enumerate(results):
        x = 28 + index * 208
        if kind == "node":
            svg.rect((x, 376, 196, 34), "node", 5)
            svg.text(x + 12, 397, name, "node-name")
            svg.text(x + 184, 397, tag, "chip-tag", "end")
        else:
            svg.chip((x, 376, 196, 34), name, tag)
    svg.text(
        28,
        436,
        "Send 4 counts send 1 because send 1 is exactly 60 minutes earlier. "
        "Sends 2 and 3 share a minute,",
        "note",
    )
    svg.text(
        28,
        454,
        "so neither counts the other. The same rule holds for the 24 hour and "
        "7 day windows.",
        "note",
    )
    return svg


def snapshot() -> Svg:
    """Draw how Snapshot(d) feeds the encoder and edge head for day d."""
    svg = Svg(
        "snapshot",
        "Snapshot and scoring",
        "Snapshot d holds earlier days only. Accounts active on day d are "
        "sampled from it "
        "and encoded, and the edge head scores each Transaction of day d.",
        880,
        420,
    )
    svg.header(
        "Snapshot(d) and scoring",
        "The GraphSAGE GNN scores day d from earlier days. No day-d edge reaches "
        "a message.",
    )
    svg.box(
        (28, 90, 184, 104),
        "History window",
        ["days before d - lag", "at most history_days", "never includes day d"],
        "focal",
    )
    svg.box(
        (246, 90, 184, 104),
        "Snapshot(d)",
        [
            "each Transaction twice:",
            "forward and reverse edge",
            "node features: counts,",
            "USD sums and maxima",
        ],
    )
    svg.box(
        (464, 90, 184, 104),
        "Sampling",
        ["NeighborLoader", "seeds: Accounts active", "on day d; gnn.neighbors"],
    )
    svg.box(
        (682, 90, 170, 104),
        "GraphSAGE encoder",
        ["layers set by", "gnn.encoder; one", "embedding per Account"],
        "node",
    )
    svg.edge([(212, 142), (246, 142)])
    svg.edge([(430, 142), (464, 142)])
    svg.edge([(648, 142), (682, 142)])
    svg.chip((464, 276, 184, 40), "Transactions of day d")
    svg.box(
        (682, 252, 170, 104),
        "Edge head",
        ["sender, receiver and", "Transaction attributes", "gnn_score, gnn_emb_*"],
        "node",
    )
    svg.edge([(556, 276), (556, 194)])
    svg.text(566, 240, "ACTIVE ACCOUNTS", "elabel")
    svg.edge([(648, 296), (682, 296)])
    svg.edge([(767, 194), (767, 252)])
    svg.box(
        (28, 252, 380, 104),
        "Training uses the same snapshots",
        [
            "LinkNeighborLoader over day-d seeds",
            "every positive + neg_sample_rate of negatives",
            "fixed epochs, no early stopping",
            "days with an empty snapshot are skipped",
        ],
    )
    return svg


def model() -> Svg:
    """Draw the GraphSAGE encoder layers and the edge head."""
    svg = Svg(
        "model",
        "GraphSAGE GNN model",
        "A node encoder feeds GraphSAGE layers, and gnn.encoder picks their type. "
        "With edge_sage, each message carries the edge encoding. With sage, "
        "messages carry neighbor states only. The Account embeddings and the edge "
        "encoding feed an edge head that outputs embedding_dim values and one logit.",
        880,
        430,
    )
    svg.header(
        "GraphSAGE encoder and edge head",
        "Sizes come from the gnn config: hidden, layers and embedding_dim.",
    )
    svg.box(
        (28, 90, 204, 100),
        "Node encoder",
        [
            "12 node features",
            "sent, received: count,",
            "USD sum, USD max",
            "Linear + ReLU",
        ],
    )
    svg.box(
        (28, 252, 204, 112),
        "Edge encoder",
        [
            "11 edge attributes",
            "7 continuous, direction,",
            "3 category codes",
            "embedded, Linear + ReLU",
        ],
    )
    svg.box(
        (290, 90, 316, 150),
        "GraphSAGE layer x layers",
        [
            "edge_sage: per-edge message = Linear of",
            "neighbor state + edge encoding, separate",
            "weights for forward and reverse edges",
            "sage: PyG SAGEConv, neighbor states only",
            "both: mean over sampled neighbors, plus",
            "Linear of own state, then ReLU",
        ],
        "node",
    )
    svg.chip((646, 140, 170, 36), "Account embeddings")
    svg.box(
        (610, 252, 242, 126),
        "Edge head",
        [
            "concat sender, receiver,",
            "edge encoding (3 x hidden)",
            "Linear + ReLU: embedding_dim",
            "Linear: one logit",
            "-> gnn_emb_*, gnn_score",
        ],
        "focal",
    )
    svg.edge([(232, 140), (290, 140)])
    svg.edge([(232, 290), (364, 290), (364, 240)], "edge-soft")
    svg.text(246, 282, "EDGE_SAGE ONLY", "elabel")
    svg.edge([(232, 346), (610, 346)])
    svg.text(372, 338, "ENCODING OF THE SCORED ROW", "elabel")
    svg.edge([(606, 158), (646, 158)])
    svg.edge([(731, 176), (731, 252)])
    return svg


def labels() -> Svg:
    """Draw which code may read which labels."""
    svg = Svg(
        "labels",
        "Label access",
        "Three loaders read the prepared table. Only report reads test and Tail "
        "labels, through its own reader.",
        880,
        380,
    )
    svg.header(
        "Who may read which labels",
        "The test labels stay unopened until report asks for them.",
    )
    svg.box(
        (28, 140, 196, 100),
        "transactions.parquet",
        ["every row, every split", "with is_laundering"],
    )
    rows = [
        (
            70,
            "split.labelled_rows",
            "train and val",
            "gnn fits, train-xgb, tune",
            "box",
            "edge",
            "the only stages that train",
        ),
        (
            158,
            "split.unlabelled_rows",
            "label column dropped",
            "features, graphs, scoring",
            "box",
            "edge",
            "none of them reads a label",
        ),
        (
            246,
            "report holdout reader",
            "test or tail, report only",
            "report only",
            "focal",
            "edge-blue",
            "the only holdout reader",
        ),
    ]
    sources = [165, 190, 215]
    for (y, call, arg, user, kind, style, note), source in zip(
        rows, sources, strict=True
    ):
        svg.box((300, y, 270, 64), call, [arg], "node" if kind == "focal" else "box")
        svg.box((618, y, 234, 64), user, [note], kind)
        mid = y + 32
        svg.edge(
            [(224, source), (262, source), (262, mid), (300, mid)]
            if source != mid
            else [(224, source), (300, mid)],
            style,
        )
        svg.edge([(570, mid), (618, mid)], style)
    svg.text(
        28,
        346,
        "split.labelled_rows raises HoldoutAccessError for test or Tail labels.",
        "note",
    )
    return svg


DIAGRAMS = {
    "pipeline": pipeline,
    "split": split,
    "crossfit": crossfit,
    "windows": windows,
    "snapshot": snapshot,
    "model": model,
    "labels": labels,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Write every diagram as an SVG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/img/readme"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, build in DIAGRAMS.items():
        svg = build()
        (args.out / f"{name}.svg").write_text(svg.render(), encoding="utf-8")
        for warning in svg.warnings:
            logger.warning(warning)
        logger.info("wrote %s", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
