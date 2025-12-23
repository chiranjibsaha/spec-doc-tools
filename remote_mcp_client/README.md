# Remote Spec MCP Proxy

FastMCP server that runs locally but forwards every tool call to a remote `spec_doc_tools` FastAPI deployment. Configure the remote API host/port/scheme in `mcp_config.json`.

## Files
- `mcp_config.json` – remote API location (`api_host`, `api_port`, `api_scheme`, optional `api_base_path`).
- `config.py` – config loader helpers.
- `client.py` – thin HTTP client for the remote API.
- `server.py` – FastMCP server publishing the tools with optional on-the-fly file writes.

## Run
```bash
python -m remote_mcp_client.server --transport streamable-http --host 0.0.0.0 --port 8820 --path /spec-remote-mcp
```
Pass `--config /path/to/mcp_config.json` to point at a different remote endpoint.

## Persisting outputs without feeding them to the LLM
Each tool accepts `persist: bool = False`. When set to `True`, the tool returns a minimal metadata payload plus a `write_to_files` array. The orchestrator should immediately call the local `write_to_files` tool with that array, writing under `tmp/spec_extracts/` (relative to the calling agent’s project root). Large markdown/JSON bodies stay only in `write_to_files`, keeping the LLM response small.

Path conventions:
- Section (chunked, no “v2” in name): `tmp/spec_extracts/{spec_id}/sections/{section_id}.md`
- Section by heading: `tmp/spec_extracts/{spec_id}/sections/{heading_slug}.md`
- Table: `tmp/spec_extracts/{spec_id}/tables/{table_id}.md`
- TOC: `tmp/spec_extracts/{spec_id}/toc.json`
- Grep results: `tmp/spec_extracts/{spec_id}/grep/{pattern_slug}.json`
- Version resolver (spec_id + presence): `tmp/spec_extracts/{spec_number}/resolve.json`

## Tool overview and curl examples (remote API)
- `spec_section_get` → `/v2/specs/{spec_id}/sections/{section_id}`  
  Example: `curl "http://<api_host>:<port>/v2/specs/38901-j10/sections/4-7-2?chunk_size=1200"`
- `spec_sections_by_heading_get` → `/specs/{spec_id}/sections/by-heading`  
  Example: `curl "http://<api_host>:<port>/specs/38901-j10/sections/by-heading?heading_text=Random%20access"`
- `spec_tables_get` → `/specs/{spec_id}/tables/{table_id}` (table_id may be provided with or without the `Table` prefix)  
  Example (no prefix): `curl "http://<api_host>:<port>/specs/38901-j10/tables/5.4-1"`
- `spec_version_resolve_get` (supports `version=latest`) → `/v2/specs/resolve?spec_number=38901&version=latest`  
  Example: `curl "http://<api_host>:<port>/v2/specs/resolve?spec_number=38901&version=latest"`
- `spec_toc_get` → `/specs/{spec_id}/toc`  
  Example: `curl "http://<api_host>:<port>/specs/38901-j10/toc?depth=3"`
- `spec_grep_get` → `/specs/{spec_id}/grep?pattern=...&regex=bool`  
  Example: `curl "http://<api_host>:<port>/specs/38901-j10/grep?pattern=beamforming&regex=false"`

## Agent prompt snippet (to maintain the write_to_files chain)
Add this to your agent instructions so the MCP payloads are written locally instead of flooding the LLM:

```
When an MCP tool response contains `write_to_files`, immediately call the local `write_to_files` tool with that array. Do not echo or summarize file contents. The MCP server already picked paths under tmp/spec_extracts/ relative to this project root.
```
