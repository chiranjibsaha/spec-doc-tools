"""Helpers for inspecting spec-document TOCs and extracting HTML sections."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from bs4 import BeautifulSoup, NavigableString, Tag

from .spec_config import load_spec_config

class SpecDocError(RuntimeError):
    """Raised for spec-document lookup/extraction errors."""


@dataclass(frozen=True, slots=True)
class TocEntry:
    clause_id: str | None
    clause_title: str
    level: int
    html_id: str
    children: Sequence["TocEntry"]


@dataclass(frozen=True, slots=True)
class TableExtract:
    table_id: str
    caption: str | None
    html: str


def table_html_to_markdown(table_html: str) -> str:
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if not table:
        return html_fragment_to_markdown(table_html)
    return _table_tag_to_markdown(table)


def _coerce_doc_basename(doc: str | Path) -> str:
    doc_str = str(doc)
    if doc_str.endswith(".html"):
        return Path(doc_str).stem
    if doc_str.endswith("_toc.json"):
        return Path(doc_str).name[: -len("_toc.json")]
    return Path(doc_str).name


def resolve_doc_paths(doc: str | Path, docs_dir: Path | None = None) -> tuple[Path, Path]:
    """Resolve the HTML and TOC JSON paths for a spec doc.

    `doc` may be a basename like "38331-i60" or a path to the HTML/TOC JSON.
    """

    if docs_dir is None:
        try:
            config = load_spec_config()
        except Exception as exc:  # pragma: no cover - unexpected config failures
            raise SpecDocError(f"Failed to load spec config: {exc}") from exc
        docs_root = config.specs_dir
    else:
        docs_root = docs_dir
    base = _coerce_doc_basename(doc)

    html_path = Path(doc)
    if html_path.suffix != ".html":
        html_path = docs_root / f"{base}.html"

    toc_path = Path(doc)
    if toc_path.name.endswith("_toc.json"):
        toc_path = toc_path
    else:
        toc_path = docs_root / f"{base}_toc.json"

    return html_path, toc_path


def load_toc_json(toc_path: Path) -> dict:
    if not toc_path.exists():
        raise SpecDocError(f"TOC JSON not found: {toc_path}")
    try:
        return json.loads(toc_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SpecDocError(f"Invalid TOC JSON: {toc_path}: {err}") from err


def _parse_toc_entry(node: dict) -> TocEntry:
    children = tuple(_parse_toc_entry(child) for child in node.get("children", []) or [])
    return TocEntry(
        clause_id=node.get("clause_id"),
        clause_title=node.get("clause_title") or "",
        level=int(node.get("level") or 0),
        html_id=node.get("id") or "",
        children=children,
    )


def load_toc_entries(toc_json: dict) -> Sequence[TocEntry]:
    toc_nodes = toc_json.get("table_of_contents")
    if not isinstance(toc_nodes, list):
        raise SpecDocError("TOC JSON missing 'table_of_contents' list")
    return tuple(_parse_toc_entry(node) for node in toc_nodes)


def iter_toc_entries(entries: Sequence[TocEntry]) -> Iterator[tuple[int, TocEntry]]:
    """Yield (depth, entry) for a TOC tree."""

    def walk(depth: int, nodes: Sequence[TocEntry]) -> Iterator[tuple[int, TocEntry]]:
        for node in nodes:
            yield depth, node
            if node.children:
                yield from walk(depth + 1, node.children)

    yield from walk(0, entries)


def _normalize_clause_id(clause_id: str) -> str:
    return clause_id.strip().rstrip(".").lower()


_SECTION_PREFIX_RE = re.compile(r"^[0-9]+(?:[-.][0-9a-z]+)+$", re.IGNORECASE)


def _looks_like_clause_id(section: str) -> bool:
    value = section.strip()
    if "." in value and _SECTION_PREFIX_RE.match(value.replace(".", "-")):
        return True
    return False


def _normalize_section_prefix(section: str) -> str:
    value = section.strip().lower()
    value = value.rstrip(".")
    value = value.replace(".", "-")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^0-9a-z-]+", "", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value


def find_toc_matches(entries: Sequence[TocEntry], section: str) -> list[TocEntry]:
    """Find TOC entries matching a clause id, full HTML id, or an HTML id prefix."""

    wanted = section.strip()
    if not wanted:
        return []

    wanted_lower = wanted.lower()
    normalized_prefix = _normalize_section_prefix(wanted)
    dotted_from_prefix = normalized_prefix.replace("-", ".")

    matches: list[TocEntry] = []
    for _, entry in iter_toc_entries(entries):
        clause_id_norm = _normalize_clause_id(entry.clause_id) if entry.clause_id else None
        if clause_id_norm and clause_id_norm == _normalize_clause_id(wanted):
            matches.append(entry)
            continue
        if entry.html_id.lower() == wanted_lower:
            matches.append(entry)
            continue
        if entry.html_id.lower() == normalized_prefix:
            matches.append(entry)
            continue
        if entry.html_id.lower().startswith(f"{normalized_prefix}-"):
            matches.append(entry)
            continue
        if clause_id_norm and clause_id_norm == dotted_from_prefix:
            matches.append(entry)
            continue
    return matches


def _normalize_table_id(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("table", "")
    return re.sub(r"[^0-9a-z_.-]", "", cleaned)


def resolve_section_html_id(entries: Sequence[TocEntry], section: str) -> str:
    """Resolve a user section selector to an HTML heading id."""

    matches = find_toc_matches(entries, section)
    if not matches:
        raise SpecDocError(f"Section not found in TOC: {section}")

    wanted = section.strip().lower()
    for entry in matches:
        if entry.html_id.lower() == wanted:
            return entry.html_id

    normalized_prefix = _normalize_section_prefix(section)
    dotted_from_prefix = normalized_prefix.replace("-", ".")
    for entry in matches:
        if entry.clause_id and _normalize_clause_id(entry.clause_id) == dotted_from_prefix:
            return entry.html_id

    if len(matches) == 1:
        return matches[0].html_id

    # Prefer the shortest html_id that still matches the prefix (e.g. 4-7-1-... over 4-7-1-1-...)
    def score(entry: TocEntry) -> tuple[int, int]:
        return (len(entry.html_id), entry.level)

    best = min(matches, key=score)
    if sum(1 for m in matches if score(m) == score(best)) > 1:
        raise SpecDocError(
            "Ambiguous section selector; provide a full HTML id or exact clause id. "
            f"Got {section}, matches: {', '.join(sorted({m.html_id for m in matches})[:10])}"
        )
    return best.html_id


_TABLE_CAPTION_RE = re.compile(r"\btable\s+(?P<id>[0-9][0-9a-z_.-]+)", flags=re.IGNORECASE)


def _extract_table_id_from_text(text: str) -> str | None:
    match = _TABLE_CAPTION_RE.search(text)
    if not match:
        return None
    return _normalize_table_id(match.group("id"))


_HEADING_RE = re.compile(
    r"<h(?P<level>[1-6])\b(?P<attrs>[^>]*)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ID_ATTR_RE = re.compile(r"""\bid\s*=\s*(?P<q>["'])(?P<id>[^"']+)(?P=q)""", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class HeadingMatch:
    level: int
    html_id: str
    start: int
    open_end: int
    close_end: int


def _iter_headings(html: str) -> Iterator[HeadingMatch]:
    for match in _HEADING_RE.finditer(html):
        attrs = match.group("attrs") or ""
        id_match = _ID_ATTR_RE.search(attrs)
        if not id_match:
            continue
        level = int(match.group("level"))
        html_id = id_match.group("id")
        close_match = re.search(rf"</h{level}\s*>", html[match.end() :], flags=re.IGNORECASE)
        if not close_match:
            # Malformed but still allow section extraction boundaries.
            close_end = match.end()
        else:
            close_end = match.end() + close_match.end()
        yield HeadingMatch(level=level, html_id=html_id, start=match.start(), open_end=match.end(), close_end=close_end)


def extract_section_html(
    html_path: Path,
    section_html_id: str,
    *,
    include_heading: bool = True,
) -> str:
    if not html_path.exists():
        raise SpecDocError(f"HTML not found: {html_path}")

    html = html_path.read_text(encoding="utf-8", errors="replace")
    headings = list(_iter_headings(html))
    if not headings:
        raise SpecDocError(f"No headings with ids found in: {html_path}")

    wanted = section_html_id.strip()
    wanted_lower = wanted.lower()

    target: HeadingMatch | None = None
    for heading in headings:
        if heading.html_id.lower() == wanted_lower:
            target = heading
            break
    if target is None:
        # Allow prefix lookups (e.g. "4-7-1" -> "4-7-1-architecture")
        prefix = _normalize_section_prefix(wanted)
        prefix_matches = [h for h in headings if h.html_id.lower() == prefix or h.html_id.lower().startswith(f"{prefix}-")]
        if not prefix_matches:
            raise SpecDocError(f"Heading id not found: {section_html_id}")

        def numeric_tail_count(heading: HeadingMatch) -> int:
            html_id = heading.html_id.lower()
            if html_id == prefix:
                return 0
            remainder = html_id[len(prefix) :]
            if remainder.startswith("-"):
                remainder = remainder[1:]
            count = 0
            while remainder:
                m = re.match(r"^(?P<num>[0-9]+)(?:-(?P<rest>.*))?$", remainder)
                if not m:
                    break
                count += 1
                remainder = m.group("rest") or ""
            return count

        prefix_matches.sort(key=lambda h: (numeric_tail_count(h), h.start))
        target = prefix_matches[0]

    start = target.start if include_heading else target.close_end
    end = len(html)
    for heading in headings:
        if heading.start <= target.start:
            continue
        if heading.level <= target.level:
            end = heading.start
            break

    return html[start:end].strip()


def _caption_text(node: Tag | NavigableString | None) -> str | None:
    if not isinstance(node, Tag):
        return None
    if node.name != "p":
        return None
    text = node.get_text(" ", strip=True)
    if text.lower().startswith("table"):
        return text
    return None


def _find_caption_for_table(table: Tag) -> str | None:
    def scan_siblings(node: Tag) -> str | None:
        for sibling in node.previous_siblings:
            if isinstance(sibling, NavigableString) and not str(sibling).strip():
                continue
            caption = _caption_text(sibling)
            if caption:
                return caption
            if isinstance(sibling, Tag) and sibling.name not in {"p", "table", "div"}:
                break
        for sibling in node.next_siblings:
            if isinstance(sibling, NavigableString) and not str(sibling).strip():
                continue
            caption = _caption_text(sibling)
            if caption:
                return caption
            if isinstance(sibling, Tag) and sibling.name not in {"p", "table", "div"}:
                break
        return None

    caption = scan_siblings(table)
    if caption:
        return caption
    # If the table is wrapped (e.g., <div class="tab"><table>...</table></div>), scan around the parent too.
    if table.parent and isinstance(table.parent, Tag):
        return scan_siblings(table.parent)
    return None


def extract_table_html(html_path: Path, table_id: str) -> TableExtract:
    """Extract a table (HTML) and its caption text from a spec document."""

    if not html_path.exists():
        raise SpecDocError(f"HTML not found: {html_path}")

    wanted = _normalize_table_id(table_id)
    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    candidates: list[tuple[Tag, str | None]] = []
    for table in soup.find_all("table"):
        caption = _find_caption_for_table(table)
        caption_id = _extract_table_id_from_text(caption or "")
        table_attr_id = _normalize_table_id(table.get("id") or "") if table.has_attr("id") else None

        if table_attr_id and table_attr_id == wanted:
            candidates.append((table, caption))
            continue
        if caption_id and caption_id == wanted:
            candidates.append((table, caption))
            continue

    if not candidates:
        raise SpecDocError(f"Table not found: {table_id}")

    if len(candidates) > 1:
        # Choose the candidate with a matching caption first, then the earliest in document order.
        def score(item: tuple[Tag, str | None]) -> tuple[int, int]:
            _, cap = item
            has_caption_match = 0 if cap and _extract_table_id_from_text(cap) == wanted else 1
            # bs4 doesn't expose start offset; rely on enumeration order.
            return (has_caption_match, 0)

        candidates.sort(key=score)

    table, caption = candidates[0]
    return TableExtract(table_id=table_id, caption=caption, html=str(table))


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", flags=re.IGNORECASE | re.DOTALL)
_BLOCK_BREAK_RE = re.compile(r"</(p|div|li|tr|h[1-6])\s*>", flags=re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def html_fragment_to_text(html_fragment: str) -> str:
    """Convert a small HTML fragment to readable text (lossy)."""

    cleaned = _SCRIPT_STYLE_RE.sub("", html_fragment)
    cleaned = _BR_RE.sub("\n", cleaned)
    cleaned = _BLOCK_BREAK_RE.sub("\n", cleaned)
    cleaned = _TAG_RE.sub("", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _table_tag_to_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    header_row: list[str] | None = None

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        texts = [html_fragment_to_text(str(cell)) for cell in cells]
        if any(cell.name == "th" for cell in cells) and header_row is None:
            header_row = texts
        else:
            rows.append(texts)

    if header_row is None:
        if not rows:
            return ""
        header_row, *data_rows = rows
    else:
        data_rows = rows

    column_count = max((len(header_row), *(len(r) for r in data_rows)), default=len(header_row))

    def pad(row: list[str]) -> list[str]:
        return row + [""] * (column_count - len(row))

    header_line = "| " + " | ".join(pad(header_row)) + " |"
    separator_line = "| " + " | ".join("---" for _ in range(column_count)) + " |"
    body_lines = ["| " + " | ".join(pad(r)) + " |" for r in data_rows]

    return "\n".join([header_line, separator_line, *body_lines])


def html_fragment_to_markdown(html_fragment: str) -> str:
    """Convert a small HTML fragment to Markdown (best-effort, lossy)."""

    cleaned = _SCRIPT_STYLE_RE.sub("", html_fragment)
    soup = BeautifulSoup(cleaned, "html.parser")

    table_placeholders: dict[str, str] = {}
    for idx, table in enumerate(soup.find_all("table")):
        md_table = _table_tag_to_markdown(table)
        placeholder = f"@@TABLE{idx}@@"
        table_placeholders[placeholder] = md_table
        table.replace_with(NavigableString(placeholder))

    cleaned = str(soup)

    # Links: <a href="...">text</a> -> [text](...)
    cleaned = re.sub(
        r"""<a\b[^>]*\bhref\s*=\s*(?P<q>["'])(?P<href>[^"']+)(?P=q)[^>]*>(?P<txt>.*?)</a\s*>""",
        lambda m: f"[{html_fragment_to_text(m.group('txt')).strip()}]({m.group('href')})",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Inline formatting
    cleaned = re.sub(r"</?(strong|b)\b[^>]*>", "**", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?(em|i)\b[^>]*>", "*", cleaned, flags=re.IGNORECASE)

    # Inline code
    cleaned = re.sub(
        r"<code\b[^>]*>(?P<code>.*?)</code\s*>",
        lambda m: f"`{html_fragment_to_text(m.group('code')).strip()}`",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Fenced code blocks: <pre>...</pre>
    def pre_repl(match: re.Match) -> str:
        inner = match.group("pre")
        inner_text = html_fragment_to_text(inner)
        return f"\n\n```\n{inner_text}\n```\n\n"

    cleaned = re.sub(r"<pre\b[^>]*>(?P<pre>.*?)</pre\s*>", pre_repl, cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Headings
    def heading_repl(match: re.Match) -> str:
        level = int(match.group("level"))
        inner = match.group("inner")
        text = html_fragment_to_text(inner).strip()
        hashes = "#" * max(1, min(6, level))
        return f"\n\n{hashes} {text}\n\n"

    cleaned = re.sub(
        r"<h(?P<level>[1-6])\b[^>]*>(?P<inner>.*?)</h(?P=level)\s*>",
        heading_repl,
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Lists (very simple)
    cleaned = re.sub(r"</?(ul|ol)\b[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"<li\b[^>]*>(?P<li>.*?)</li\s*>",
        lambda m: f"- {html_fragment_to_text(m.group('li')).strip()}\n",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Paragraphs and breaks
    cleaned = _BR_RE.sub("\n", cleaned)
    cleaned = re.sub(r"</p\s*>", "\n\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<p\b[^>]*>", "", cleaned, flags=re.IGNORECASE)

    # Drop remaining tags and normalize whitespace.
    cleaned = _TAG_RE.sub("", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    for placeholder, md_table in table_placeholders.items():
        cleaned = cleaned.replace(placeholder, f"\n\n{md_table}\n\n")

    return cleaned.strip()


def filter_toc_entries(
    entries: Sequence[TocEntry],
    *,
    prefix: str | None = None,
    search: str | None = None,
    max_depth: int | None = None,
) -> list[tuple[int, TocEntry]]:
    prefix_norm = _normalize_section_prefix(prefix) if prefix else None
    search_norm = search.strip().lower() if search else None

    results: list[tuple[int, TocEntry]] = []
    for depth, entry in iter_toc_entries(entries):
        if max_depth is not None and depth > max_depth:
            continue
        if prefix_norm:
            clause_id_norm = _normalize_clause_id(entry.clause_id) if entry.clause_id else None
            if clause_id_norm and not clause_id_norm.startswith(prefix_norm.replace("-", ".")):
                if not entry.html_id.lower().startswith(prefix_norm):
                    continue
            elif not clause_id_norm and not entry.html_id.lower().startswith(prefix_norm):
                continue
        if search_norm:
            hay = f"{entry.clause_id or ''} {entry.clause_title} {entry.html_id}".lower()
            if search_norm not in hay:
                continue
        results.append((depth, entry))
    return results


def format_toc_as_text(items: Iterable[tuple[int, TocEntry]]) -> str:
    lines: list[str] = []
    for depth, entry in items:
        indent = "  " * depth
        clause = entry.clause_id or "-"
        title = entry.clause_title.replace("\n", " ").strip()
        title = re.sub(r"\s+", " ", title)
        lines.append(f"{indent}{clause}\t{title}\t#{entry.html_id}")
    return "\n".join(lines)
