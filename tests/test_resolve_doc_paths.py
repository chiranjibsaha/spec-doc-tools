from pathlib import Path

from spec_doc_tools.spec_docs import resolve_doc_paths


def _write_spec(tmp_path: Path, spec_id: str, corrected: bool = False) -> None:
    spec_dir = tmp_path / spec_id
    spec_dir.mkdir(parents=True, exist_ok=True)

    html_name = f"image_corrected_{spec_id}.html" if corrected else f"{spec_id}.html"
    (spec_dir / html_name).write_text("<html></html>", encoding="utf-8")
    (spec_dir / f"{spec_id}_toc.json").write_text("{}", encoding="utf-8")


def test_resolve_prefers_corrected_html_when_present(tmp_path: Path) -> None:
    spec_id = "38901-j10"
    _write_spec(tmp_path, spec_id, corrected=True)
    # also write the standard html to ensure preference order
    (tmp_path / spec_id / f"{spec_id}.html").write_text("<html></html>", encoding="utf-8")

    html_path, toc_path = resolve_doc_paths(spec_id, docs_dir=tmp_path)

    assert html_path.name == f"image_corrected_{spec_id}.html"
    assert toc_path.name == f"{spec_id}_toc.json"


def test_resolve_falls_back_to_default_html(tmp_path: Path) -> None:
    spec_id = "38901-j10"
    _write_spec(tmp_path, spec_id, corrected=False)

    html_path, toc_path = resolve_doc_paths(spec_id, docs_dir=tmp_path)

    assert html_path.name == f"{spec_id}.html"
    assert toc_path.name == f"{spec_id}_toc.json"
