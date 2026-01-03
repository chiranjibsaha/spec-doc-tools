from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.grep_result import GrepResult
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    spec_id: str,
    *,
    pattern: str,
    regex: bool | Unset = False,
    docs_dir: None | str | Unset = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["pattern"] = pattern

    params["regex"] = regex

    json_docs_dir: None | str | Unset
    if isinstance(docs_dir, Unset):
        json_docs_dir = UNSET
    else:
        json_docs_dir = docs_dir
    params["docs_dir"] = json_docs_dir

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/specs/{spec_id}/grep".format(
            spec_id=quote(str(spec_id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> GrepResult | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = GrepResult.from_dict(response.json())

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
) -> Response[GrepResult | HTTPValidationError]:
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
    pattern: str,
    regex: bool | Unset = False,
    docs_dir: None | str | Unset = UNSET,
) -> Response[GrepResult | HTTPValidationError]:
    """Grep Spec

     Search a spec HTML document for a substring and return structured matches.

    Args:
        spec_id (str):
        pattern (str): Substring to search (case-insensitive).
        regex (bool | Unset): Treat pattern as a regex (case-insensitive). Invalid regex returns
            HTTP 400. Default: False.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GrepResult | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        spec_id=spec_id,
        pattern=pattern,
        regex=regex,
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
    pattern: str,
    regex: bool | Unset = False,
    docs_dir: None | str | Unset = UNSET,
) -> GrepResult | HTTPValidationError | None:
    """Grep Spec

     Search a spec HTML document for a substring and return structured matches.

    Args:
        spec_id (str):
        pattern (str): Substring to search (case-insensitive).
        regex (bool | Unset): Treat pattern as a regex (case-insensitive). Invalid regex returns
            HTTP 400. Default: False.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GrepResult | HTTPValidationError
    """

    return sync_detailed(
        spec_id=spec_id,
        client=client,
        pattern=pattern,
        regex=regex,
        docs_dir=docs_dir,
    ).parsed


async def asyncio_detailed(
    spec_id: str,
    *,
    client: AuthenticatedClient | Client,
    pattern: str,
    regex: bool | Unset = False,
    docs_dir: None | str | Unset = UNSET,
) -> Response[GrepResult | HTTPValidationError]:
    """Grep Spec

     Search a spec HTML document for a substring and return structured matches.

    Args:
        spec_id (str):
        pattern (str): Substring to search (case-insensitive).
        regex (bool | Unset): Treat pattern as a regex (case-insensitive). Invalid regex returns
            HTTP 400. Default: False.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GrepResult | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        spec_id=spec_id,
        pattern=pattern,
        regex=regex,
        docs_dir=docs_dir,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    spec_id: str,
    *,
    client: AuthenticatedClient | Client,
    pattern: str,
    regex: bool | Unset = False,
    docs_dir: None | str | Unset = UNSET,
) -> GrepResult | HTTPValidationError | None:
    """Grep Spec

     Search a spec HTML document for a substring and return structured matches.

    Args:
        spec_id (str):
        pattern (str): Substring to search (case-insensitive).
        regex (bool | Unset): Treat pattern as a regex (case-insensitive). Invalid regex returns
            HTTP 400. Default: False.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GrepResult | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            spec_id=spec_id,
            client=client,
            pattern=pattern,
            regex=regex,
            docs_dir=docs_dir,
        )
    ).parsed
