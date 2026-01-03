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
- Change port: `spec-api --port 9000`  
- Change host: `spec-api --host 127.0.0.1`  
Set the specs root via `--docs-dir /path/to/specs`, or export `SPEC_DOCS_DIR`. When installed site-wide, you can also point to a config file with `SPEC_CONFIG_PATH=/path/to/spec_config.json`.

Endpoints (+ curl examples):
- Sections v2: `GET /v2/specs/{spec_id}/sections/{section_ref}` — Markdown + images (single chunk; `chunk_size` is ignored).  
  Example: `curl "http://localhost:8010/v2/specs/38901-j10/sections/4-7-2"`
- Section summary: `GET /v2/specs/{spec_id}/sections/{section_ref}/summary` — Returns `chars`/`bytes` only (no content).  
  Example: `curl "http://localhost:8010/v2/specs/38901-j10/sections/4-7-2/summary"`
- Sections by heading: `GET /specs/{spec_id}/sections/by-heading?heading_text=...` — Case-insensitive heading match.  
  Example: `curl "http://localhost:8010/specs/38901-j10/sections/by-heading?heading_text=Random%20access"`
- Tables: `GET /specs/{spec_id}/tables/{table_id}` — Table to Markdown with caption. `table_id` can be passed with or without the `Table` prefix.  
  Example (no prefix): `curl "http://localhost:8010/specs/38901-j10/tables/5.4-1"`
- Version resolver (supports `version=latest` or `major/minor/patch`): `GET /v2/specs/resolve?spec_number=38901&version=latest`  
  Example: `curl "http://localhost:8010/v2/specs/resolve?spec_number=38901&version=latest"`
- TOC: `GET /specs/{spec_id}/toc` — Optional `depth` and `section_ref` filters.  
  - `section_ref`: full heading id to resolve and return the section body text (heading excluded).  
  Examples:  
  - `curl "http://localhost:8010/specs/38901-j10/toc?depth=3"`  
  - `curl "http://localhost:8010/specs/38901-j10/toc?section_ref=7-4-3-1-o2i-building-penetration-loss"` -> includes `html_id` and `section_text` for that heading.
- Grep: `GET /specs/{spec_id}/grep?pattern=...&regex=bool` — Substring/regex search.  
  Example: `curl "http://localhost:8010/specs/38901-j10/grep?pattern=beamforming&regex=false"`
- `GET /health`, `GET /help` (tool metadata).

Example v2 response (single chunk):
```json
{
  "status": "ok",
  "spec_id": "38901-j10",
  "section_id": "4-7-2",
  "html_id": "4-7-2-protocol-stacks",
  "include_heading": true,
  "markdown": {
    "bytes": 8267,
    "md": "### 4.7.2 ... full markdown ...",
    "chunk_count": 1,
    "chunk_size": 8267,
    "chunks": [
      {
        "index": 1,
        "bytes": 8267,
        "md_snippet": "### 4.7.2 ... full markdown ...",
        "images": []
      }
    ]
  },
  "images": [],
  "source": {
    "html_path": "/path/to/38901-j10.html",
    "toc_path": "/path/to/38901-j10_toc.json"
  }
}
```

## FastMCP
Expose the same API as MCP tools:
```bash
spec-mcp --host 0.0.0.0 --port 8810 --path /spec-mcp  # default transport: streamable-http
```
Tools (operation ids): `spec_sections_get`, `spec_sections_by_heading_get`, `spec_tables_get`, `spec_toc_get`, `spec_grep_get`. The `/help` endpoint is published as the `spec_help` MCP resource. `spec_health_get` is excluded from tools.

## Files expected
New layout (per spec id under docs root). Example for `spec_id = 38901-j10`:
- HTML: `docs/38901-j10/38901-j10.html` (if present, `docs/38901-j10/image_corrected_38901-j10.html` is preferred automatically)
- TOC JSON: `docs/38901-j10/38901-j10_toc.json`
- Images: `docs/38901-j10/images/`

## Notes
- `docs_dir` overrides the configured specs directory on every CLI/API/MCP call.
- Grep supports `regex=true` with `re.IGNORECASE` semantics; invalid regex → HTTP 400.
