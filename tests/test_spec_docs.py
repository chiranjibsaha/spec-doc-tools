from __future__ import annotations

from pathlib import Path

import pytest

from spec_doc_tools.spec_docs import SpecDocError, extract_table_html, resolve_doc_paths


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolve_doc_paths_prefers_nested(monkeypatch, tmp_path: Path) -> None:
    """Nested layout is chosen when both nested and flat exist."""

    root = tmp_path / "specs"
    nested = root / "38300-j00"
    flat_html = root / "38300-j00.html"
    flat_toc = root / "38300-j00_toc.json"
    nested_html = nested / "38300-j00.html"
    nested_toc = nested / "38300-j00_toc.json"

    # Create both nested and flat; nested should win.
    _write(flat_html, "<html></html>")
    _write(flat_toc, '{"table_of_contents": []}')
    _write(nested_html, "<html></html>")
    _write(nested_toc, '{"table_of_contents": []}')

    monkeypatch.setenv("SPEC_DOCS_DIR", str(root))

    html_path, toc_path = resolve_doc_paths("38300-j00")
    assert html_path == nested_html
    assert toc_path == nested_toc


def test_resolve_doc_paths_falls_back_to_flat(monkeypatch, tmp_path: Path) -> None:
    """Flat files are used when nested layout is absent."""

    root = tmp_path / "specs"
    flat_html = root / "38300-j00.html"
    flat_toc = root / "38300-j00_toc.json"
    _write(flat_html, "<html></html>")
    _write(flat_toc, '{"table_of_contents": []}')

    monkeypatch.setenv("SPEC_DOCS_DIR", str(root))

    html_path, toc_path = resolve_doc_paths("38300-j00")
    assert html_path == flat_html
    assert toc_path == flat_toc


def test_extract_table_accepts_id_without_prefix(tmp_path: Path) -> None:
    """Table lookup should work when caller omits the 'Table' prefix."""

    html_file = tmp_path / "doc.html"
    table_html = """
    <html>
      <body>
        <p>Table 5.4-1: Caption text</p>
        <table id="Table5.4-1">
          <tr><th>A</th><th>B</th></tr>
          <tr><td>1</td><td>2</td></tr>
        </table>
      </body>
    </html>
    """
    _write(html_file, table_html)

    extracted = extract_table_html(html_file, "5.4-1")
    assert extracted.table_id == "5.4-1"
    assert "Caption text" in (extracted.caption or "")
    assert "<table" in extracted.html.lower()


def test_extract_table_raises_on_missing(tmp_path: Path) -> None:
    html_file = tmp_path / "doc.html"
    _write(html_file, "<html><body>No tables here</body></html>")

    with pytest.raises(SpecDocError):
        extract_table_html(html_file, "5.4-1")
