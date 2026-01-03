from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.toc_response import TOCResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    spec_id: str,
    *,
    depth: int | None | Unset = UNSET,
    section_ref: None | str | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_depth: int | None | Unset
    if isinstance(depth, Unset):
        json_depth = UNSET
    else:
        json_depth = depth
    params["depth"] = json_depth

    json_section_ref: None | str | Unset
    if isinstance(section_ref, Unset):
        json_section_ref = UNSET
    else:
        json_section_ref = section_ref
    params["section_ref"] = json_section_ref

    json_docs_dir: None | str | Unset
    if isinstance(docs_dir, Unset):
        json_docs_dir = UNSET
    else:
        json_docs_dir = docs_dir
    params["docs_dir"] = json_docs_dir

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/specs/{spec_id}/toc".format(
            spec_id=quote(str(spec_id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TOCResponse | None:
    if response.status_code == 200:
        response_200 = TOCResponse.from_dict(response.json())

        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[HTTPValidationError | TOCResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    spec_id: str,
    *,
    client: AuthenticatedClient | Client,
    depth: int | None | Unset = UNSET,
    section_ref: None | str | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TOCResponse]:
    """Get Toc

    Args:
        spec_id (str):
        depth (int | None | Unset): Limit to this heading depth (1=top level). Applies to tree
            depth in the TOC.
        section_ref (None | str | Unset): Full heading id; when provided, also return the section
            text under that heading.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TOCResponse]
    """

    kwargs = _get_kwargs(
        spec_id=spec_id,
        depth=depth,
        section_ref=section_ref,
        docs_dir=docs_dir,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    spec_id: str,
    *,
    client: AuthenticatedClient | Client,
    depth: int | None | Unset = UNSET,
    section_ref: None | str | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> HTTPValidationError | TOCResponse | None:
    """Get Toc

    Args:
        spec_id (str):
        depth (int | None | Unset): Limit to this heading depth (1=top level). Applies to tree
            depth in the TOC.
        section_ref (None | str | Unset): Full heading id; when provided, also return the section
            text under that heading.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TOCResponse
    """

    return sync_detailed(
        spec_id=spec_id,
        client=client,
        depth=depth,
        section_ref=section_ref,
        docs_dir=docs_dir,
    ).parsed


async def asyncio_detailed(
    spec_id: str,
    *,
    client: AuthenticatedClient | Client,
    depth: int | None | Unset = UNSET,
    section_ref: None | str | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TOCResponse]:
    """Get Toc

    Args:
        spec_id (str):
        depth (int | None | Unset): Limit to this heading depth (1=top level). Applies to tree
            depth in the TOC.
        section_ref (None | str | Unset): Full heading id; when provided, also return the section
            text under that heading.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TOCResponse]
    """

    kwargs = _get_kwargs(
        spec_id=spec_id,
        depth=depth,
        section_ref=section_ref,
        docs_dir=docs_dir,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    spec_id: str,
    *,
    client: AuthenticatedClient | Client,
    depth: int | None | Unset = UNSET,
    section_ref: None | str | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> HTTPValidationError | TOCResponse | None:
    """Get Toc

    Args:
        spec_id (str):
        depth (int | None | Unset): Limit to this heading depth (1=top level). Applies to tree
            depth in the TOC.
        section_ref (None | str | Unset): Full heading id; when provided, also return the section
            text under that heading.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TOCResponse
    """

    return (
        await asyncio_detailed(
            spec_id=spec_id,
            client=client,
            depth=depth,
            section_ref=section_ref,
            docs_dir=docs_dir,
        )
    ).parsed
