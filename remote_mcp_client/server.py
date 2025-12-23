"""FastMCP server that proxies spec_doc_tools APIs to a remote host."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from fastmcp import FastMCP

from . import __version__
from .client import RemoteSpecApiClient
from .config import MCPConfig, load_mcp_config

INSTRUCTIONS = """MCP proxy that forwards spec_doc_tools calls to a remote FastAPI server.
Configure the remote host/port/scheme in mcp_config.json.
Tools mirror the upstream API: sections, tables, toc, grep, version resolution, and help."""


def _build_server(config: MCPConfig) -> FastMCP:
    client = RemoteSpecApiClient(config)
    name = "spec-remote-proxy"
    mcp = FastMCP(
        name=name,
        version=__version__,
        instructions=INSTRUCTIONS,
        tags={"spec_doc_tools", "remote-proxy"},
    )

    def _slug(text: str, fallback: str = "item") -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", "-", text)
        text = re.sub(r"[^a-z0-9._-]", "", text)
        return text or fallback

    def _rel_path(*parts: str) -> str:
        return str(Path("tmp") / "spec_extracts" / Path(*parts))

    def _persist_content(content: str, rel_path: str, base: dict, persist: bool) -> dict:
        """Attach write_to_files payload and remove bulky fields when persisting."""
        if persist:
            base = dict(base)  # shallow copy
            base["write_to_files"] = [{"path": rel_path, "content": content}]
            # Avoid flooding the LLM with large payloads.
            base["content_path"] = rel_path
        return base

    @mcp.tool(description="Extract a section with chunking and embedded images (stored locally when persist=true).")
    def spec_section_get(
        spec_id: str,
        section_id: str,
        include_heading: bool = True,
        chunk_size: int | None = None,
        docs_dir: str | None = None,
        persist: bool = False,
    ) -> dict:
        resp = client.spec_sections_v2_get(
            spec_id=spec_id,
            section_id=section_id,
            include_heading=include_heading,
            chunk_size=chunk_size,
            docs_dir=docs_dir,
        )
        md = resp.get("markdown", {}).get("md", "")
        rel_path = _rel_path(spec_id, "sections", f"{section_id}.md")
        base = {
            "status": resp.get("status", "ok"),
            "spec_id": spec_id,
            "section_id": section_id,
            "html_id": resp.get("html_id"),
            "include_heading": include_heading,
            "bytes": len(md.encode("utf-8")),
            "chunk_count": resp.get("markdown", {}).get("chunk_count"),
            "chunk_size": resp.get("markdown", {}).get("chunk_size"),
        }
        return _persist_content(md, rel_path, base, persist)

    @mcp.tool(description="Resolve spec number + version to spec_id and file presence.")
    def spec_version_resolve_get(
        spec_number: str,
        version: str | None = None,
        major: int | None = None,
        minor: int | None = None,
        patch: int | None = None,
        docs_dir: str | None = None,
        persist: bool = False,
    ) -> dict:
        resp = client.spec_version_resolve_get(
            spec_number,
            version=version,
            major=major,
            minor=minor,
            patch=patch,
            docs_dir=docs_dir,
        )
        rel_path = _rel_path(spec_number, "resolve.json")
        base = {
            "status": resp.get("status", "ok"),
            "spec_number": spec_number,
            "version": resp.get("version"),
            "spec_id": resp.get("spec_id"),
            "exists": resp.get("exists"),
        }
        return _persist_content(json.dumps(resp), rel_path, base, persist)

    @mcp.tool(description="Find a section by heading text (case-insensitive).")
    def spec_sections_by_heading_get(
        spec_id: str,
        heading_text: str,
        include_heading: bool = True,
        docs_dir: str | None = None,
        persist: bool = False,
    ) -> dict:
        resp = client.spec_sections_by_heading_get(
            spec_id,
            heading_text,
            include_heading=include_heading,
            docs_dir=docs_dir,
        )
        md = resp.get("markdown", {}).get("md", "")
        filename = f"{_slug(heading_text)}.md"
        rel_path = _rel_path(spec_id, "sections", filename)
        base = {
            "status": resp.get("status", "ok"),
            "spec_id": spec_id,
            "section_heading": heading_text,
            "html_id": resp.get("html_id"),
            "include_heading": include_heading,
            "bytes": len(md.encode("utf-8")),
        }
        return _persist_content(md, rel_path, base, persist)

    @mcp.tool(description="Extract a table to markdown with caption.")
    def spec_tables_get(spec_id: str, table_id: str, docs_dir: str | None = None, persist: bool = False) -> dict:
        resp = client.spec_tables_get(spec_id, table_id, docs_dir=docs_dir)
        md = resp.get("markdown", {}).get("md", "")
        rel_path = _rel_path(spec_id, "tables", f"{table_id}.md")
        base = {
            "status": resp.get("status", "ok"),
            "spec_id": spec_id,
            "table_id": table_id,
            "caption": resp.get("caption"),
            "bytes": len(md.encode("utf-8")),
        }
        return _persist_content(md, rel_path, base, persist)

    @mcp.tool(description="Return TOC entries with optional depth/section filters.")
    def spec_toc_get(
        spec_id: str,
        depth: int | None = None,
        section_id: str | None = None,
        docs_dir: str | None = None,
        persist: bool = False,
    ) -> dict:
        resp = client.spec_toc_get(spec_id, depth=depth, section_id=section_id, docs_dir=docs_dir)
        rel_path = _rel_path(spec_id, "toc.json")
        content = resp
        base = {
            "status": resp.get("status", "ok"),
            "spec_id": spec_id,
            "depth_limit": depth,
            "section_filter": section_id,
            "items": len(resp.get("toc", []) or []),
        }
        return _persist_content(json.dumps(content), rel_path, base, persist)

    @mcp.tool(description="Search spec HTML with substring or regex.")
    def spec_grep_get(
        spec_id: str,
        pattern: str,
        regex: bool = False,
        docs_dir: str | None = None,
        persist: bool = False,
    ) -> dict:
        resp = client.spec_grep_get(spec_id, pattern, regex=regex, docs_dir=docs_dir)
        rel_path = _rel_path(spec_id, "grep", f"{_slug(pattern)}.json")
        base = {
            "status": resp.get("status", "ok"),
            "spec_id": spec_id,
            "pattern": pattern,
            "regex": regex,
            "match_count": resp.get("match_count", 0),
        }
        return _persist_content(json.dumps(resp), rel_path, base, persist)

    @mcp.tool(description="Health check for the remote API.")
    def spec_health_get() -> dict:
        return client.spec_health_get()

    @mcp.resource(description="List available tools and shapes from the remote API.")
    def spec_help() -> dict:
        return client.spec_help_get()

    return mcp


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="spec-remote-mcp",
        description="Expose remote spec_doc_tools API over FastMCP.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to mcp_config.json (defaults to remote_mcp_client/mcp_config.json).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http", "sse"],
        default="streamable-http",
        help="FastMCP transport to use (streamable-http works with Codex MCP).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for HTTP/SSE transports (ignored for stdio).",
    )
    parser.add_argument(
        "--port",
        default=8820,
        type=int,
        help="Port for HTTP/SSE transports.",
    )
    parser.add_argument(
        "--path",
        default="/spec-remote-mcp",
        help="HTTP path for streamable-http/SSE transports.",
    )
    args = parser.parse_args(argv)

    config = load_mcp_config(args.config)
    server = _build_server(config)

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
        )


__all__ = ["main"]
