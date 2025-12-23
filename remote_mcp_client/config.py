"""Configuration loader for the remote Spec MCP proxy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = {
    "api_host": "127.0.0.1",
    "api_port": 8010,
    "api_scheme": "http",
    "api_base_path": "",
}

CONFIG_FILENAME = "mcp_config.json"


@dataclass(frozen=True, slots=True)
class MCPConfig:
    api_host: str
    api_port: int
    api_scheme: str
    api_base_path: str

    @property
    def base_url(self) -> str:
        host_port = f"{self.api_host}:{self.api_port}"
        base_path = self.api_base_path.strip()
        if base_path and not base_path.startswith("/"):
            base_path = f"/{base_path}"
        base_path = base_path.rstrip("/")
        return f"{self.api_scheme}://{host_port}{base_path}"


def _load_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid MCP config JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"MCP config must be a JSON object: {path}")
    return data


def load_mcp_config(config_path: Path | None = None) -> MCPConfig:
    """Load MCP proxy configuration (host/port/scheme/base path)."""

    path = config_path or Path(__file__).resolve().parent / CONFIG_FILENAME
    merged = DEFAULT_CONFIG.copy()
    if path.exists():
        merged.update({k: v for k, v in _load_json(path).items() if v is not None})

    scheme = merged["api_scheme"].lower()
    if scheme not in {"http", "https"}:
        raise RuntimeError("api_scheme must be 'http' or 'https'")

    try:
        port = int(merged["api_port"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("api_port must be an integer") from exc

    return MCPConfig(
        api_host=str(merged["api_host"]),
        api_port=port,
        api_scheme=scheme,
        api_base_path=str(merged.get("api_base_path", "")),
    )


__all__ = ["MCPConfig", "load_mcp_config", "CONFIG_FILENAME"]
