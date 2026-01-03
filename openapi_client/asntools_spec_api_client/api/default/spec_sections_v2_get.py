from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.spec_sections_v2_get_response_spec_sections_v2_get import SpecSectionsV2GetResponseSpecSectionsV2Get
from ...types import UNSET, Response, Unset


def _get_kwargs(
    spec_id: str,
    section_id: str,
    *,
    include_heading: bool | Unset = True,
    chunk_size: int | Unset = 1200,
    docs_dir: None | str | Unset = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["include_heading"] = include_heading

    params["chunk_size"] = chunk_size

    json_docs_dir: None | str | Unset
    if isinstance(docs_dir, Unset):
        json_docs_dir = UNSET
    else:
        json_docs_dir = docs_dir
    params["docs_dir"] = json_docs_dir

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v2/specs/{spec_id}/sections/{section_id}".format(
            spec_id=quote(str(spec_id), safe=""),
            section_id=quote(str(section_id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get | None:
    if response.status_code == 200:
        response_200 = SpecSectionsV2GetResponseSpecSectionsV2Get.from_dict(response.json())

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
) -> Response[HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    spec_id: str,
    section_id: str,
    *,
    client: AuthenticatedClient | Client,
    include_heading: bool | Unset = True,
    chunk_size: int | Unset = 1200,
    docs_dir: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get]:
    """Get Section V2

     Extract a section as Markdown with embedded images (base64).

    Args:
        spec_id (str):
        section_id (str):
        include_heading (bool | Unset): Include the heading tag in the extraction. Default: True.
        chunk_size (int | Unset): Ignored (kept for backward compatibility). Default: 1200.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get]
    """

    kwargs = _get_kwargs(
        spec_id=spec_id,
        section_id=section_id,
        include_heading=include_heading,
        chunk_size=chunk_size,
        docs_dir=docs_dir,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    spec_id: str,
    section_id: str,
    *,
    client: AuthenticatedClient | Client,
    include_heading: bool | Unset = True,
    chunk_size: int | Unset = 1200,
    docs_dir: None | str | Unset = UNSET,
) -> HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get | None:
    """Get Section V2

     Extract a section as Markdown with embedded images (base64).

    Args:
        spec_id (str):
        section_id (str):
        include_heading (bool | Unset): Include the heading tag in the extraction. Default: True.
        chunk_size (int | Unset): Ignored (kept for backward compatibility). Default: 1200.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get
    """

    return sync_detailed(
        spec_id=spec_id,
        section_id=section_id,
        client=client,
        include_heading=include_heading,
        chunk_size=chunk_size,
        docs_dir=docs_dir,
    ).parsed


async def asyncio_detailed(
    spec_id: str,
    section_id: str,
    *,
    client: AuthenticatedClient | Client,
    include_heading: bool | Unset = True,
    chunk_size: int | Unset = 1200,
    docs_dir: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get]:
    """Get Section V2

     Extract a section as Markdown with embedded images (base64).

    Args:
        spec_id (str):
        section_id (str):
        include_heading (bool | Unset): Include the heading tag in the extraction. Default: True.
        chunk_size (int | Unset): Ignored (kept for backward compatibility). Default: 1200.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get]
    """

    kwargs = _get_kwargs(
        spec_id=spec_id,
        section_id=section_id,
        include_heading=include_heading,
        chunk_size=chunk_size,
        docs_dir=docs_dir,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    spec_id: str,
    section_id: str,
    *,
    client: AuthenticatedClient | Client,
    include_heading: bool | Unset = True,
    chunk_size: int | Unset = 1200,
    docs_dir: None | str | Unset = UNSET,
) -> HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get | None:
    """Get Section V2

     Extract a section as Markdown with embedded images (base64).

    Args:
        spec_id (str):
        section_id (str):
        include_heading (bool | Unset): Include the heading tag in the extraction. Default: True.
        chunk_size (int | Unset): Ignored (kept for backward compatibility). Default: 1200.
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SpecSectionsV2GetResponseSpecSectionsV2Get
    """

    return (
        await asyncio_detailed(
            spec_id=spec_id,
            section_id=section_id,
            client=client,
            include_heading=include_heading,
            chunk_size=chunk_size,
            docs_dir=docs_dir,
        )
    ).parsed
