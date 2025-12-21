"""FastMCP server exposing the spec_doc_tools API as MCP tools."""
from __future__ import annotations

import argparse
from typing import Sequence

from fastmcp import FastMCP
from fastmcp.server.openapi.routing import MCPType, RouteMap

from . import __version__
from .spec_server import app as fastapi_app

INSTRUCTIONS = """MCP interface for spec_doc_tools (sections/tables/toc/grep over spec HTML).
Use:
- spec_sections_get: extract a section by id as markdown with chunk metadata.
- spec_sections_by_heading_get: extract a section by matching heading text.
- spec_tables_get: extract a table by caption/id to markdown with chunking.
- spec_toc_get: fetch the TOC tree with optional depth/section filter.
- spec_grep_get: search spec HTML with substring or regex, returning match metadata.
Read the `spec_doc_tools` help docs or call the `spec_help` resource for details."""

OPERATION_NAME_MAP = {
    "spec_sections_get": "spec_sections_get",
    "spec_sections_by_heading_get": "spec_sections_by_heading_get",
    "spec_tables_get": "spec_tables_get",
    "spec_toc_get": "spec_toc_get",
    "spec_grep_get": "spec_grep_get",
    "spec_health_get": "spec_health_get",
    "spec_help_get": "spec_help",
}

ROUTE_MAPS: Sequence[RouteMap] = (
    RouteMap(
        methods=["GET"],
        pattern=r"^/health$",
        mcp_type=MCPType.EXCLUDE,
    ),
    RouteMap(
        methods=["GET"],
        pattern=r"^/help$",
        mcp_type=MCPType.RESOURCE,
    ),
)


def build_spec_fastmcp_server() -> FastMCP:
    """Create a FastMCP server backed by the spec FastAPI app."""

    return FastMCP.from_fastapi(
        fastapi_app,
        name="asntools-spec",
        version=__version__,
        instructions=INSTRUCTIONS,
        route_maps=list(ROUTE_MAPS),
        mcp_names=OPERATION_NAME_MAP,
        tags={"spec_doc_tools"},
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="asntools-spec-mcp",
        description="Expose spec_doc_tools over the FastMCP protocol.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind when using HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--port",
        default=8810,
        type=int,
        help="Port for HTTP/SSE transports.",
    )
    parser.add_argument(
        "--path",
        default="/spec-mcp",
        help="HTTP path for streamable-http/SSE transports (default: /spec-mcp).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http", "sse"],
        default="streamable-http",
        help="FastMCP transport to use (streamable-http works with Codex MCP).",
    )
    args = parser.parse_args(argv)

    server = build_spec_fastmcp_server()
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
        )


__all__ = ["build_spec_fastmcp_server", "main"]
