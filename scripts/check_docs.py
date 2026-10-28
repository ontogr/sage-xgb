"""Check relative links, anchors, image sources, and code blocks in the docs.

The check covers the Markdown files in the repository root and every Markdown
file under docs/. It reports relative targets that do not exist, `#anchor`
fragments that match no heading, and fenced `python` blocks that do not parse
or are never closed. It uses only the standard library.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

_CODE_SPAN_PATTERN = re.compile(r"(?<!`)(`+)(?!`).+?(?<!`)\1(?!`)")
_LINK_PATTERN = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_REFERENCE_DEFINITION_PATTERN = re.compile(r"^ {0,3}\[(?!\^)[^\]]+\]:\s*<?([^\s>]+)")
_SOURCE_ATTRIBUTE_PATTERN = re.compile(r"(?<![\w-])(src|srcset)\s*=\s*([\"'])(.*?)\2")
_FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_HEADING_PATTERN = re.compile(r"^ {0,3}#{1,6}\s+(.*?)(?:\s+#+)?\s*$")
_HEADING_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_ANCHOR_STRIP_PATTERN = re.compile(r"[^\w\- ]")
_URL_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


@dataclass(frozen=True)
class Problem:
    """One failed check, located by file and line.

    Attributes:
        path: Markdown file, relative to the repository root.
        line: 1-based line number in that file.
        message: What is wrong and which value caused it.
    """

    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        """Return the location and the message on one line."""
        return f"{self.path.as_posix()}:{self.line}: {self.message}"


@dataclass(frozen=True)
class _CodeBlock:
    """A fenced code block and the line where its body starts.

    Attributes:
        info: Text after the opening fence, such as `python`.
        body: Lines between the fences.
        first_line: 1-based line number of the first body line.
        closed: Whether a closing fence was found.
    """

    info: str
    body: str
    first_line: int
    closed: bool


def _split_markdown(text: str) -> tuple[list[tuple[int, str]], list[_CodeBlock]]:
    """Return the prose lines outside fences and the fenced code blocks.

    A fence that is never closed runs to the end of the text and is returned
    with `closed` set to False.
    """
    prose: list[tuple[int, str]] = []
    blocks: list[_CodeBlock] = []
    fence: str | None = None
    info = ""
    body: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), start=1):
        match = _FENCE_PATTERN.match(line)
        if fence is None:
            if match is None:
                prose.append((number, line))
                continue
            fence = match.group(1)
            info = match.group(2).strip()
            body = []
            start = number + 1
            continue
        if (
            match is not None
            and match.group(1).startswith(fence)
            and not match.group(2).strip()
        ):
            blocks.append(_CodeBlock(info, "\n".join(body), start, closed=True))
            fence = None
            continue
        body.append(line)
    if fence is not None:
        blocks.append(_CodeBlock(info, "\n".join(body), start, closed=False))
    return prose, blocks


def _heading_anchor(heading: str) -> str:
    """Return the GitHub-style anchor for heading text.

    Link syntax keeps only its text. Characters other than letters, digits,
    underscores, hyphens, and spaces are removed, and each space becomes a
    hyphen. Repeated headings are not numbered here.
    """
    text = _HEADING_LINK_PATTERN.sub(r"\1", heading).lower()
    return _ANCHOR_STRIP_PATTERN.sub("", text).replace(" ", "-")


def _heading_anchors(path: Path) -> set[str]:
    """Return the anchors of the headings in a Markdown file.

    A repeated heading gets a `-1`, `-2`, ... suffix, as GitHub renders it.
    Headings inside fenced code blocks are ignored.
    """
    prose, _ = _split_markdown(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    anchors: set[str] = set()
    for _, line in prose:
        match = _HEADING_PATTERN.match(line)
        if match is None:
            continue
        anchor = _heading_anchor(match.group(1))
        repeat = counts[anchor]
        counts[anchor] += 1
        anchors.add(anchor if repeat == 0 else f"{anchor}-{repeat}")
    return anchors


def _line_targets(line: str) -> list[str]:
    """Return the link, reference definition, and src or srcset targets in a line.

    Code spans are removed first, so link-shaped text inside them is ignored.
    A srcset value contributes the URL of each candidate.
    """
    text = _CODE_SPAN_PATTERN.sub("", line)
    targets = [match.group(1) for match in _LINK_PATTERN.finditer(text)]
    definition = _REFERENCE_DEFINITION_PATTERN.match(text)
    if definition is not None:
        targets.append(definition.group(1))
    for match in _SOURCE_ATTRIBUTE_PATTERN.finditer(text):
        value = match.group(3)
        if match.group(1) == "srcset":
            targets.extend(
                candidate.split()[0]
                for candidate in value.split(",")
                if candidate.strip()
            )
        else:
            targets.append(value)
    return targets


def _target_problems(
    source: Path, root: Path, line_number: int, target: str
) -> Iterator[Problem]:
    """Yield a problem when a relative target or its `#anchor` does not resolve.

    Targets with a URL scheme or a leading `//` are skipped. A bare `#anchor`
    refers to the source file. An anchor is checked only when the target is a
    Markdown file.
    """
    if _URL_SCHEME_PATTERN.match(target) or target.startswith("//"):
        return
    file_part, _, anchor = target.partition("#")
    destination = source
    if file_part:
        destination = source.parent / file_part
        if not destination.exists():
            yield Problem(
                source.relative_to(root),
                line_number,
                f"link target does not exist: {target}",
            )
            return
    if not anchor or destination.suffix != ".md":
        return
    if anchor not in _heading_anchors(destination):
        yield Problem(
            source.relative_to(root),
            line_number,
            f"anchor does not match a heading: {target}",
        )


def _link_problems(
    path: Path, root: Path, prose: Sequence[tuple[int, str]]
) -> Iterator[Problem]:
    """Yield a problem for each broken link, reference definition, or source."""
    for line_number, line in prose:
        for target in _line_targets(line):
            yield from _target_problems(path, root, line_number, target)


def _syntax_problems(
    path: Path, root: Path, blocks: Sequence[_CodeBlock]
) -> Iterator[Problem]:
    """Yield a problem for each unclosed fence and each `python` block that fails.

    A `python` block that does not parse is located by its line in the file.
    """
    for block in blocks:
        if not block.closed:
            yield Problem(
                path.relative_to(root),
                block.first_line - 1,
                "code fence is never closed",
            )
            continue
        if block.info.split()[:1] != ["python"]:
            continue
        try:
            ast.parse(block.body)
        except SyntaxError as error:
            line = block.first_line + (error.lineno or 1) - 1
            yield Problem(
                path.relative_to(root),
                line,
                f"python block does not parse: {error.msg}",
            )


def _markdown_files(root: Path) -> list[Path]:
    """Return the Markdown files in the root, then every one under docs/."""
    candidates = [*sorted(root.glob("*.md")), *sorted((root / "docs").rglob("*.md"))]
    return [candidate for candidate in candidates if candidate.is_file()]


def check_docs(root: Path) -> list[Problem]:
    """Return every broken link, anchor, source, and unparsable Python block.

    Args:
        root: Repository root that holds the Markdown files and docs/.

    Returns:
        The problems found, in file order. An empty list means the check passed.
    """
    problems: list[Problem] = []
    for path in _markdown_files(root):
        prose, blocks = _split_markdown(path.read_text(encoding="utf-8"))
        problems.extend(_link_problems(path, root, prose))
        problems.extend(_syntax_problems(path, root, blocks))
    return problems


def main() -> int:
    """Run the check on the repository and return the process exit code.

    Problems go to standard error, one per line. The exit code is 0 when the
    check passes and 1 otherwise.
    """
    problems = check_docs(REPO_ROOT)
    if problems:
        for problem in problems:
            sys.stderr.write(f"{problem}\n")
        sys.stderr.write(f"docs check failed with {len(problems)} problem(s)\n")
        return 1
    print(f"docs check passed for {len(_markdown_files(REPO_ROOT))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
