# Agent Integration Notes

When wiring this MCP server into an agent, ensure the write-to-files handoff is honored so large payloads bypass the LLM and get saved locally.

Prompt snippet (adjust paths if your project root differs):

```
When an MCP tool response contains `write_to_files`, immediately call the local `write_to_files` tool with that array. Do not show or summarize the file contents. The MCP server already chose the destination paths under `tmp/spec_extracts/` relative to this project root.
```

Key expectations:
- Always pass through `write_to_files` without editing content.
- Keep the LLM output minimal; rely on the metadata in the MCP response.
- Files are intended to live under `tmp/spec_extracts/` in the agent’s workspace.
