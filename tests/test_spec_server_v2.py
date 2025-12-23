from pathlib import Path

from fastapi.testclient import TestClient

from spec_doc_tools.spec_server import app


def _write_fixture_spec(tmp_path: Path, spec_id: str) -> None:
    spec_dir = tmp_path / spec_id
    spec_dir.mkdir(parents=True, exist_ok=True)

    html = (
        "<html><body>"
        '<h2 id="4-7-2">Clause 4.7.2</h2>'
        "<p>Short body text.</p>"
        "</body></html>"
    )
    (spec_dir / f"{spec_id}.html").write_text(html, encoding="utf-8")

    toc = {
        "table_of_contents": [
            {
                "clause_id": "4.7.2",
                "clause_title": "Clause 4.7.2",
                "level": 2,
                "id": "4-7-2",
                "children": [],
            }
        ]
    }
    (spec_dir / f"{spec_id}_toc.json").write_text(
        __import__("json").dumps(toc), encoding="utf-8"
    )


def test_section_v2_returns_single_chunk(tmp_path: Path) -> None:
    spec_id = "38901-j10"
    _write_fixture_spec(tmp_path, spec_id)

    client = TestClient(app)
    resp = client.get(
        f"/v2/specs/{spec_id}/sections/4-7-2",
        params={"docs_dir": str(tmp_path)},
    )

    assert resp.status_code == 200
    payload = resp.json()

    markdown = payload["markdown"]
    assert markdown["chunk_count"] == 1
    assert len(markdown["chunks"]) == 1
    assert markdown["chunks"][0]["md_snippet"]  # full content present
