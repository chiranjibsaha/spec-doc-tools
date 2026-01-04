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


def _write_custom_heading_spec(tmp_path: Path, spec_id: str, heading_id: str, clause_id: str, title: str) -> None:
    spec_dir = tmp_path / spec_id
    spec_dir.mkdir(parents=True, exist_ok=True)

    html = (
        "<html><body>"
        f'<h3 id="{heading_id}">{title}</h3>'
        "<p>Body for the special heading.</p>"
        "</body></html>"
    )
    (spec_dir / f"{spec_id}.html").write_text(html, encoding="utf-8")

    toc = {
        "table_of_contents": [
            {
                "clause_id": clause_id,
                "clause_title": title,
                "level": 3,
                "id": heading_id,
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
    resp = client.get(f"/v1/specs/{spec_id}/sections/4-7-2", params={"docs_dir": str(tmp_path)})

    assert resp.status_code == 200
    payload = resp.json()

    markdown = payload["markdown"]
    assert markdown["chunk_count"] == 1
    assert len(markdown["chunks"]) == 1
    chunk = markdown["chunks"][0]
    assert chunk["md_snippet"]  # full content present
    assert chunk["bytes"] == markdown["bytes"]


def test_toc_returns_tree_only(tmp_path: Path) -> None:
    spec_id = "38901-j10"
    _write_fixture_spec(tmp_path, spec_id)

    client = TestClient(app)
    resp = client.get(f"/v1/specs/{spec_id}/toc", params={"docs_dir": str(tmp_path)})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["spec_id"] == spec_id
    assert payload["toc"]
    assert payload["toc"][0]["clause_id_ref"]
    assert "section_ref" not in payload
    assert "section_text" not in payload
    assert "html_id" not in payload


def test_toc_defaults_to_latest_when_no_suffix(tmp_path: Path) -> None:
    spec_number = "38901"
    old_id = f"{spec_number}-j09"
    new_id = f"{spec_number}-j10"
    _write_fixture_spec(tmp_path, old_id)
    _write_custom_heading_spec(
        tmp_path,
        new_id,
        heading_id="4-7-2",
        clause_id="4.7.2",
        title="Clause 4.7.2",
    )

    client = TestClient(app)
    resp = client.get(f"/v1/specs/{spec_number}/toc", params={"docs_dir": str(tmp_path)})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["spec_id"] == new_id  # latest suffix picked
    assert payload["toc"]


def test_toc_accepts_dotted_spec_number(tmp_path: Path) -> None:
    spec_number_dot = "38.901"
    base_number = "38901"
    latest_id = f"{base_number}-j10"
    _write_fixture_spec(tmp_path, f"{base_number}-j09")
    _write_fixture_spec(tmp_path, latest_id)

    client = TestClient(app)
    resp = client.get(f"/v1/specs/{spec_number_dot}/toc", params={"docs_dir": str(tmp_path)})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["spec_id"] == latest_id
    assert payload["toc"]


def test_toc_uses_version_query_when_provided(tmp_path: Path) -> None:
    spec_number = "38901"
    target_id = f"{spec_number}-j00"
    _write_fixture_spec(tmp_path, target_id)

    client = TestClient(app)
    resp = client.get(
        f"/v1/specs/{spec_number}/toc",
        params={"docs_dir": str(tmp_path), "version": "19.0.0"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["spec_id"] == target_id
