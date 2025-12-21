"""Configuration helpers for spec-document tools."""

from __future__ import annotations

import json
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

    cfg_path = config_path or _project_root() / CONFIG_FILENAME
    if cfg_path.exists():
        data = _load_config_json(cfg_path)
        specs_dir_raw = data.get("specs_dir") or DEFAULT_SPEC_DIR
    else:
        specs_dir_raw = DEFAULT_SPEC_DIR

    specs_dir = Path(specs_dir_raw)
    if not specs_dir.is_absolute():
        specs_dir = (cfg_path.parent / specs_dir).resolve()

    return SpecConfig(specs_dir=specs_dir)
