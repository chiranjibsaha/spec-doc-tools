"""Configuration helpers for spec-document tools."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SPEC_DIR = Path("doc")
CONFIG_FILENAME = "spec_config.json"


@dataclass(frozen=True, slots=True)
class SpecConfig:
    specs_dir: Path


def _project_root() -> Path:
    # asntools/ -> project root
    return Path(__file__).resolve().parent.parent


def _load_config_json(config_path: Path) -> dict:
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid spec config JSON: {config_path}: {exc}") from exc


def load_spec_config(config_path: Path | None = None) -> SpecConfig:
    """Load spec configuration from the project root, falling back to defaults."""

    # Resolve config path:
    # 1) explicit argument
    # 2) SPEC_CONFIG_PATH env
    # 3) package root / spec_config.json
    # 4) cwd / spec_config.json
    cfg_path = (
        config_path
        or (Path(os.environ["SPEC_CONFIG_PATH"]) if "SPEC_CONFIG_PATH" in os.environ else None)
        or _project_root() / CONFIG_FILENAME
    )
    if cfg_path and not cfg_path.exists():
        # Try current working directory as a fallback when installed package lacks the file.
        alt = Path.cwd() / CONFIG_FILENAME
        cfg_path = alt if alt.exists() else cfg_path

    if cfg_path and cfg_path.exists():
        data = _load_config_json(cfg_path)
        specs_dir_raw = data.get("specs_dir") or DEFAULT_SPEC_DIR
    else:
        cfg_path = None
        specs_dir_raw = DEFAULT_SPEC_DIR

    # Environment override takes highest precedence.
    if "SPEC_DOCS_DIR" in os.environ:
        specs_dir_raw = os.environ["SPEC_DOCS_DIR"]

    specs_dir = Path(specs_dir_raw)
    if not specs_dir.is_absolute():
        base = cfg_path.parent if cfg_path else Path.cwd()
        specs_dir = (base / specs_dir).resolve()

    return SpecConfig(specs_dir=specs_dir)
