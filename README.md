# Spec Doc Tools (standalone)

Tools for working with 3GPP spec HTML/TOC bundles:
- CLI: table of contents inspection and section extraction.
- API: FastAPI server for sections/tables/TOC/grep.
- MCP: FastMCP wrapper publishing the same endpoints as tools.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
q
```

Specs directory is configured via `spec_config.json` (defaults to `./doc`), or override with `--docs-dir` / `docs_dir` on API calls.

## CLI
- TOC: `spec-toc 38300-j00 [--prefix 4.7 | --search architecture | --format json]`
- Extract section: `spec-extract 38300-j00 4-7-2 --format md|text|html [--output path.md] [--out-dir dir] [--no-include-heading]`

## API (FastAPI)
Run: `spec-api  # defaults to 0.0.0.0:8010`

Endpoints:
- `GET /specs/{spec_id}/sections/{section_id}` — Markdown + chunk metadata (`bytes`, `chunk_count`, `chunks[*].md_snippet`); `chunk_size` tunes chunk length.
- `GET /v2/specs/{spec_id}/sections/{section_id}` — Same as above but also returns embedded images (base64 + paths) per chunk.
- `GET /specs/{spec_id}/sections/by-heading?heading_text=...` — Match heading text (case-insensitive); supports `include_heading` and `chunk_size`; 404 returns `suggestions`.
- `GET /specs/{spec_id}/tables/{table_id}` — Table to Markdown with caption + chunks; match by caption prefix or table id.
- `GET /v2/specs/resolve?spec_number=38901&version=19.1.0` — Build `spec_id` (or use `version=latest` or `major/minor/patch`) and report file/folder presence for the spec bundle.
- `GET /specs/{spec_id}/toc` — TOC JSON; optional `depth` limit and `section_id` prefix filter.
- `GET /specs/{spec_id}/grep?pattern=...&regex=bool` — Case-insensitive substring or regex search over spec HTML. Returns `match_count`, per-match `line`, `char_offset`, `message_length`, `message`, plus `chunks` (one per match).
- `GET /health`, `GET /help` (tool metadata).

## FastMCP
Expose the same API as MCP tools:
```bash
spec-mcp --host 0.0.0.0 --port 8810 --path /spec-mcp  # default transport: streamable-http
```
Tools (operation ids): `spec_sections_get`, `spec_sections_by_heading_get`, `spec_tables_get`, `spec_toc_get`, `spec_grep_get`. The `/help` endpoint is published as the `spec_help` MCP resource. `spec_health_get` is excluded from tools.

## Files expected
New layout (per spec id under docs root). Example for `spec_id = 38901-j10`:
- HTML: `docs/38901-j10/38901-j10.html`
- TOC JSON: `docs/38901-j10/38901-j10_toc.json`
- Images: `docs/38901-j10/images/`

## Notes
- `docs_dir` overrides the configured specs directory on every CLI/API/MCP call.
- Grep supports `regex=true` with `re.IGNORECASE` semantics; invalid regex → HTTP 400.
