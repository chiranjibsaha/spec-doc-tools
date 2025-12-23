"""Remote FastMCP proxy for spec_doc_tools API."""

from __future__ import annotations

try:
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import version  # type: ignore

try:
    __version__ = version("spec-doc-tools")
except Exception:  # pragma: no cover
    try:
        from spec_doc_tools import __version__ as _pkg_version
    except Exception:
        _pkg_version = "0.0.0"
    __version__ = _pkg_version

__all__ = ["__version__"]
