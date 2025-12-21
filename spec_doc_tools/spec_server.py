"""FastAPI server for spec document extraction (sections + tables)."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
import argparse
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .spec_docs import (
    SpecDocError,
    extract_section_html,
    extract_table_html,
    filter_toc_entries,
    html_fragment_to_markdown,
    load_toc_entries,
    load_toc_json,
    resolve_doc_paths,
    resolve_section_html_id,
    table_html_to_markdown,
)

DEFAULT_SPEC_API_PORT = 8010

app = FastAPI(title="asntools-spec-api", version=__version__)

HELP_ENTRIES = [
    {
        "name": "spec_sections_get",
        "method": "GET",
        "path": "/specs/{spec_id}/sections/{section_id}",
        "description": "Extract a section as markdown.",
        "request": {
            "path": {"spec_id": "str", "section_id": "str"},
            "query": {
                "include_heading": "bool (default true)",
                "docs_dir": "Optional[str]",
            },
        },
        "response": {
            "status": "ok",
            "spec_id": "str",
            "section_id": "str",
            "html_id": "str",
            "markdown": "dict(bytes, md)",
            "source": "dict(html_path, toc_path)",
        },
    },
    {
        "name": "spec_sections_by_heading_get",
        "method": "GET",
        "path": "/specs/{spec_id}/sections/by-heading",
        "description": "Find a section by heading text (case-insensitive) and return markdown.",
        "request": {
            "query": {
                "spec_id": "str",
                "heading_text": "str",
                "include_heading": "bool",
                "docs_dir": "Optional[str]",
            }
        },
        "response": {
            "status": "ok",
            "spec_id": "str",
            "section_heading": "str",
            "html_id": "str",
            "markdown": "dict",
            "source": "dict(html_path, toc_path)",
        },
    },
    {
        "name": "spec_tables_get",
        "method": "GET",
        "path": "/specs/{spec_id}/tables/{table_id}",
        "description": "Extract a table to markdown with caption.",
        "request": {
            "path": {"spec_id": "str", "table_id": "str"},
            "query": {"docs_dir": "Optional[str]"},
        },
        "response": {
            "status": "ok",
            "spec_id": "str",
            "table_id": "str",
            "caption": "str",
            "markdown": "dict",
            "source": "dict(html_path)",
            "html": "str",
        },
    },
    {
        "name": "spec_toc_get",
        "method": "GET",
        "path": "/specs/{spec_id}/toc",
        "description": "Return TOC entries with optional depth/section filters.",
        "request": {
            "query": {
                "spec_id": "str",
                "depth": "int?",
                "section_id": "str?",
                "docs_dir": "Optional[str]",
            }
        },
        "response": {
            "status": "ok",
            "spec_id": "str",
            "toc": "List[dict(depth, clause_id, clause_title, level, id)]",
            "source": "dict(toc_path)",
        },
    },
    {
        "name": "spec_grep_get",
        "method": "GET",
        "path": "/specs/{spec_id}/grep",
        "description": "Search spec HTML with substring or regex.",
        "request": {
            "query": {
                "spec_id": "str",
                "pattern": "str",
                "regex": "bool (default false)",
                "docs_dir": "Optional[str]",
            }
        },
        "response": {
            "status": "ok | not_found",
            "spec_id": "str",
            "query": "str",
            "match_count": "int",
            "matches": "List[dict(index,line,char_offset,message_length,message)]",
            "chunks": "List[str]",
            "source": "dict(html_path, toc_path)",
        },
    },
]


def _build_markdown(md: str) -> dict:
    """Return markdown payload (no chunking)."""
    if not md:
        md = ""
    return {"bytes": len(md.encode("utf-8")), "md": md}


_HEADING_ID_RE = re.compile(r"<h([1-6])[^>]*\bid\s*=\s*\"(?P<id>[^\"]+)\"", re.IGNORECASE)


def _clause_from_html_id(html_id: str) -> Optional[str]:
    """Extract clause id from a heading html id. Example: '5-23-1-sl-bch' -> '5.23.1'."""
    value = html_id.replace("-", ".")
    m = re.match(r"(?P<num>\d+(?:\.\d+)*)", value)
    return m.group("num") if m else None


def _search_spec_text(pattern: str, html_path: Path, use_regex: bool = False) -> dict:
    if not pattern:
        raise ValueError("pattern must be non-empty")
    if not html_path.exists():
        raise FileNotFoundError(f"Spec HTML not found at {html_path}")

    text = html_path.read_text(encoding="utf-8")
    flags = re.IGNORECASE
    if use_regex:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
    else:
        pattern_lower = pattern.lower()

    matches = []
    chunks = []
    char_offset = 0
    current_clause = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        heading_match = _HEADING_ID_RE.search(line)
        if heading_match:
            current_clause = _clause_from_html_id(heading_match.group("id"))

        line_matches = []
        if use_regex:
            for match in compiled.finditer(line):
                idx = match.start()
                line_matches.append(
                    {
                        "index": len(matches) + len(line_matches) + 1,
                        "line": line_no,
                        "char_offset": char_offset + idx,
                        "message_length": len(line),
                        "message": line,
                        "clause_id": current_clause,
                    }
                )
        else:
            search_start = 0
            line_lower = line.lower()
            while True:
                idx = line_lower.find(pattern_lower, search_start)
                if idx == -1:
                    break
                line_matches.append(
                    {
                        "index": len(matches) + len(line_matches) + 1,
                        "line": line_no,
                        "char_offset": char_offset + idx,
                        "message_length": len(line),
                        "message": line,
                        "clause_id": current_clause,
                    }
                )
                search_start = idx + len(pattern_lower) or idx + 1
        char_offset += len(line) + 1
        if line_matches:
            matches.extend(line_matches)
            chunks.append({"index": line_matches[0]["index"], "text": line, "clause_id": current_clause})

    if not matches:
        return {
            "status": "not_found",
            "match_count": 0,
            "matches": [],
            "chunks": [],
        }

    return {
        "status": "ok",
        "match_count": len(matches),
        "matches": matches,
        "chunks": [m["message"] for m in matches],
    }


def _find_heading_by_title(entries, heading_text: str):
    wanted = heading_text.strip().lower()
    # Exact (case-insensitive) match wins
    for _, entry in filter_toc_entries(entries, search=None):
        if entry.clause_title.strip().lower() == wanted:
            return entry
    # Otherwise any entry containing the substring
    candidates = [e for _, e in filter_toc_entries(entries, search=heading_text)]
    if candidates:
        candidates.sort(key=lambda e: (len(e.clause_title), len(e.html_id)))
        return candidates[0]
    return None


@app.get("/specs/{spec_id}/sections/by-heading", operation_id="spec_sections_by_heading_get")
def get_section_by_heading(
    spec_id: str,
    heading_text: str = Query(..., description="Heading text to match (case-insensitive)."),
    include_heading: bool = Query(True, description="Include the heading tag in the extraction."),
    docs_dir: Optional[str] = Query(
        None,
        description="Optional override for the specs directory. Defaults to specs_dir from spec_config.json.",
    ),
) -> dict:
    """Find a section by heading text and return it as markdown."""
    try:
        html_path, toc_path = _resolve_paths(spec_id, docs_dir)
        toc_json = load_toc_json(toc_path)
        entries = load_toc_entries(toc_json)
        entry = _find_heading_by_title(entries, heading_text)
        if not entry:
            titles = [e.clause_title for _, e in filter_toc_entries(entries, search=None)]
            suggestions = difflib.get_close_matches(heading_text, titles, n=5, cutoff=0.4)
            raise HTTPException(
                status_code=404,
                detail={"message": f"Heading not found: {heading_text}", "suggestions": suggestions},
            )
        section_html_id = entry.html_id
        fragment = extract_section_html(html_path, section_html_id, include_heading=include_heading)
        markdown = html_fragment_to_markdown(fragment)
        payload = _build_markdown(markdown)
    except SpecDocError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "spec_id": spec_id,
        "section_heading": heading_text,
        "html_id": section_html_id,
        "include_heading": include_heading,
        "markdown": payload,
        "source": {"html_path": str(html_path), "toc_path": str(toc_path)},
    }


def _resolve_paths(spec_id: str, docs_dir: Optional[str]) -> tuple[Path, Path]:
    docs_path = Path(docs_dir) if docs_dir else None
    return resolve_doc_paths(spec_id, docs_dir=docs_path)


@app.get("/specs/{spec_id}/toc", operation_id="spec_toc_get")
def get_toc(
    spec_id: str,
    depth: Optional[int] = Query(
        None,
        ge=1,
        description="Limit to this heading depth (1=top level). Applies to tree depth in the TOC.",
    ),
    section_id: Optional[str] = Query(
        None,
        description="Optional clause/html id prefix filter, e.g. '2.2.1' or '2-2-1'.",
    ),
    docs_dir: Optional[str] = Query(
        None,
        description="Optional override for the specs directory. Defaults to specs_dir from spec_config.json.",
    ),
) -> dict:
    try:
        _, toc_path = _resolve_paths(spec_id, docs_dir)
        toc_json = load_toc_json(toc_path)
        entries = load_toc_entries(toc_json)
        items = filter_toc_entries(
            entries,
            prefix=section_id,
            max_depth=(depth - 1) if depth is not None else None,
        )
    except SpecDocError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    toc_items = [
        {
            "depth": depth_val,
            "clause_id": entry.clause_id,
            "clause_title": entry.clause_title,
            "level": entry.level,
            "id": entry.html_id,
        }
        for depth_val, entry in items
    ]
    return {
        "status": "ok",
        "spec_id": spec_id,
        "depth_limit": depth,
        "section_filter": section_id,
        "toc": toc_items,
        "source": {"toc_path": str(toc_path)},
    }


@app.get("/health", operation_id="spec_health_get")
def health() -> dict:
    return {"status": "ok"}


@app.get("/help", operation_id="spec_help_get")
def help_endpoint() -> dict:
    return {"status": "ok", "tools": HELP_ENTRIES}


@app.get("/specs/{spec_id}/grep", operation_id="spec_grep_get")
def grep_spec(
    spec_id: str,
    pattern: str = Query(..., min_length=1, description="Substring to search (case-insensitive)."),
    regex: bool = Query(
        False, description="Treat pattern as a regex (case-insensitive). Invalid regex returns HTTP 400."
    ),
    docs_dir: Optional[str] = Query(
        None,
        description="Optional override for the specs directory. Defaults to specs_dir from spec_config.json.",
    ),
) -> dict:
    """Search a spec HTML document for a substring and return structured matches."""

    try:
        html_path, toc_path = _resolve_paths(spec_id, docs_dir)
        payload = _search_spec_text(pattern, html_path, use_regex=regex)
    except SpecDocError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload.update(
        {
            "spec_id": spec_id,
            "query": pattern,
            "source": {"html_path": str(html_path), "toc_path": str(toc_path)},
        }
    )
    return payload


@app.get("/specs/{spec_id}/sections/{section_id}", operation_id="spec_sections_get")
def get_section(
    spec_id: str,
    section_id: str,
    include_heading: bool = Query(True, description="Include the heading tag in the extraction."),
    docs_dir: Optional[str] = Query(
        None,
        description="Optional override for the specs directory. Defaults to specs_dir from spec_config.json.",
    ),
) -> dict:
    """Extract a section as Markdown."""

    try:
        html_path, toc_path = _resolve_paths(spec_id, docs_dir)
        toc_json = load_toc_json(toc_path)
        entries = load_toc_entries(toc_json)
        section_html_id = resolve_section_html_id(entries, section_id)
        fragment = extract_section_html(html_path, section_html_id, include_heading=include_heading)
        markdown = html_fragment_to_markdown(fragment)
        payload = _build_markdown(markdown)
    except SpecDocError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "spec_id": spec_id,
        "section_id": section_id,
        "html_id": section_html_id,
        "include_heading": include_heading,
        "markdown": payload,
        "source": {"html_path": str(html_path), "toc_path": str(toc_path)},
    }


@app.get("/specs/{spec_id}/tables/{table_id}", operation_id="spec_tables_get")
def get_table(
    spec_id: str,
    table_id: str,
    docs_dir: Optional[str] = Query(
        None,
        description="Optional override for the specs directory. Defaults to specs_dir from spec_config.json.",
    ),
) -> dict:
    """Extract a specific table as structured Markdown."""

    try:
        html_path, _ = _resolve_paths(spec_id, docs_dir)
        extracted = extract_table_html(html_path, table_id)
        markdown = table_html_to_markdown(extracted.html)
        payload = _build_markdown(markdown)
    except SpecDocError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "spec_id": spec_id,
        "table_id": table_id,
        "caption": extracted.caption,
        "markdown": payload,
        "source": {"html_path": str(html_path)},
        "html": extracted.html,
    }


def main(argv: Optional[list[str]] = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run the Spec Doc Tools FastAPI server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_SPEC_API_PORT, help=f"Port to bind (default: {DEFAULT_SPEC_API_PORT})")
    parser.add_argument(
        "--docs-dir",
        dest="docs_dir",
        help="Optional override for specs_dir; applied process-wide before startup.",
    )
    args = parser.parse_args(argv)

    if args.docs_dir:
        # Override the default specs directory for this process.
        from . import spec_config

        spec_config.DEFAULT_SPEC_DIR = Path(args.docs_dir)

    # Export the in-package app rather than expecting an external top-level module.
    uvicorn.run("spec_doc_tools.spec_server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
