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

## Agent prompt snippet (to maintain the write_to_files chain)
Add this to your agent instructions so the MCP payloads are written locally instead of flooding the LLM:

```
When an MCP tool response contains `write_to_files`, immediately call the local `write_to_files` tool with that array. Do not echo or summarize file contents. The MCP server already picked paths under tmp/spec_extracts/ relative to this project root.
```
