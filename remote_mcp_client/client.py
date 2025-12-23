"""Thin HTTP client for the remote spec_doc_tools API."""

from __future__ import annotations

from typing import Any, Mapping

import requests
from requests import Response
from requests.exceptions import RequestException

from .config import MCPConfig


def _clean_params(params: Mapping[str, Any]) -> dict:
    return {k: v for k, v in params.items() if v is not None}


class RemoteSpecApiClient:
    """Minimal wrapper around requests for the spec_doc_tools FastAPI service."""

    def __init__(self, config: MCPConfig, session: requests.Session | None = None) -> None:
        self.config = config
        self._session = session or requests.Session()

    def _request(self, method: str, path: str, params: Mapping[str, Any] | None = None) -> dict:
        url = f"{self.config.base_url}{path}"
        try:
            resp: Response = self._session.request(method, url, params=_clean_params(params or {}), timeout=30)
        except RequestException as exc:
            raise RuntimeError(f"Failed to reach remote API at {url}: {exc}") from exc

        if resp.status_code >= 400:
            raise RuntimeError(f"Remote API error {resp.status_code} for {url}: {resp.text}")
        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(f"Remote API returned non-JSON response for {url}") from exc

    # Endpoint wrappers -----------------------------------------------------

    def spec_sections_v2_get(
        self,
        spec_id: str,
        section_id: str,
        include_heading: bool = True,
        chunk_size: int | None = None,
        docs_dir: str | None = None,
    ) -> dict:
        return self._request(
            "GET",
            f"/v2/specs/{spec_id}/sections/{section_id}",
            params={
                "include_heading": include_heading,
                "chunk_size": chunk_size,
                "docs_dir": docs_dir,
            },
        )

    def spec_version_resolve_get(
        self,
        spec_number: str,
        version: str | None = None,
        major: int | None = None,
        minor: int | None = None,
        patch: int | None = None,
        docs_dir: str | None = None,
    ) -> dict:
        return self._request(
            "GET",
            "/v2/specs/resolve",
            params={
                "spec_number": spec_number,
                "version": version,
                "major": major,
                "minor": minor,
                "patch": patch,
                "docs_dir": docs_dir,
            },
        )

    def spec_sections_by_heading_get(
        self,
        spec_id: str,
        heading_text: str,
        include_heading: bool = True,
        docs_dir: str | None = None,
    ) -> dict:
        return self._request(
            "GET",
            f"/specs/{spec_id}/sections/by-heading",
            params={
                "heading_text": heading_text,
                "include_heading": include_heading,
                "docs_dir": docs_dir,
            },
        )

    def spec_tables_get(self, spec_id: str, table_id: str, docs_dir: str | None = None) -> dict:
        return self._request(
            "GET",
            f"/specs/{spec_id}/tables/{table_id}",
            params={"docs_dir": docs_dir},
        )

    def spec_toc_get(
        self,
        spec_id: str,
        depth: int | None = None,
        section_id: str | None = None,
        docs_dir: str | None = None,
    ) -> dict:
        return self._request(
            "GET",
            f"/specs/{spec_id}/toc",
            params={
                "depth": depth,
                "section_id": section_id,
                "docs_dir": docs_dir,
            },
        )

    def spec_grep_get(
        self, spec_id: str, pattern: str, regex: bool = False, docs_dir: str | None = None
    ) -> dict:
        return self._request(
            "GET",
            f"/specs/{spec_id}/grep",
            params={"pattern": pattern, "regex": regex, "docs_dir": docs_dir},
        )

    def spec_health_get(self) -> dict:
        return self._request("GET", "/health")

    def spec_help_get(self) -> dict:
        return self._request("GET", "/help")


__all__ = ["RemoteSpecApiClient"]
