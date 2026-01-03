from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.spec_version_resolve_get_response_spec_version_resolve_get import (
    SpecVersionResolveGetResponseSpecVersionResolveGet,
)
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    spec_number: str,
    version: None | str | Unset = UNSET,
    major: int | None | Unset = UNSET,
    minor: int | None | Unset = UNSET,
    patch: int | None | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    params["spec_number"] = spec_number

    json_version: None | str | Unset
    if isinstance(version, Unset):
        json_version = UNSET
    else:
        json_version = version
    params["version"] = json_version

    json_major: int | None | Unset
    if isinstance(major, Unset):
        json_major = UNSET
    else:
        json_major = major
    params["major"] = json_major

    json_minor: int | None | Unset
    if isinstance(minor, Unset):
        json_minor = UNSET
    else:
        json_minor = minor
    params["minor"] = json_minor

    json_patch: int | None | Unset
    if isinstance(patch, Unset):
        json_patch = UNSET
    else:
        json_patch = patch
    params["patch"] = json_patch

    json_docs_dir: None | str | Unset
    if isinstance(docs_dir, Unset):
        json_docs_dir = UNSET
    else:
        json_docs_dir = docs_dir
    params["docs_dir"] = json_docs_dir

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v2/specs/resolve",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet | None:
    if response.status_code == 200:
        response_200 = SpecVersionResolveGetResponseSpecVersionResolveGet.from_dict(response.json())

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
) -> Response[HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    spec_number: str,
    version: None | str | Unset = UNSET,
    major: int | None | Unset = UNSET,
    minor: int | None | Unset = UNSET,
    patch: int | None | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet]:
    """Resolve Spec Version

     Build spec_id from spec number + version, and report file/folder presence.

    Args:
        spec_number (str): Base spec number, e.g. 38901
        version (None | str | Unset): Full version string, e.g. 19.1.0 or 'latest'
        major (int | None | Unset): Major version (0-35)
        minor (int | None | Unset): Minor version (0-9)
        patch (int | None | Unset): Patch version (0-9)
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet]
    """

    kwargs = _get_kwargs(
        spec_number=spec_number,
        version=version,
        major=major,
        minor=minor,
        patch=patch,
        docs_dir=docs_dir,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    spec_number: str,
    version: None | str | Unset = UNSET,
    major: int | None | Unset = UNSET,
    minor: int | None | Unset = UNSET,
    patch: int | None | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet | None:
    """Resolve Spec Version

     Build spec_id from spec number + version, and report file/folder presence.

    Args:
        spec_number (str): Base spec number, e.g. 38901
        version (None | str | Unset): Full version string, e.g. 19.1.0 or 'latest'
        major (int | None | Unset): Major version (0-35)
        minor (int | None | Unset): Minor version (0-9)
        patch (int | None | Unset): Patch version (0-9)
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet
    """

    return sync_detailed(
        client=client,
        spec_number=spec_number,
        version=version,
        major=major,
        minor=minor,
        patch=patch,
        docs_dir=docs_dir,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    spec_number: str,
    version: None | str | Unset = UNSET,
    major: int | None | Unset = UNSET,
    minor: int | None | Unset = UNSET,
    patch: int | None | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet]:
    """Resolve Spec Version

     Build spec_id from spec number + version, and report file/folder presence.

    Args:
        spec_number (str): Base spec number, e.g. 38901
        version (None | str | Unset): Full version string, e.g. 19.1.0 or 'latest'
        major (int | None | Unset): Major version (0-35)
        minor (int | None | Unset): Minor version (0-9)
        patch (int | None | Unset): Patch version (0-9)
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet]
    """

    kwargs = _get_kwargs(
        spec_number=spec_number,
        version=version,
        major=major,
        minor=minor,
        patch=patch,
        docs_dir=docs_dir,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    spec_number: str,
    version: None | str | Unset = UNSET,
    major: int | None | Unset = UNSET,
    minor: int | None | Unset = UNSET,
    patch: int | None | Unset = UNSET,
    docs_dir: None | str | Unset = UNSET,
) -> HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet | None:
    """Resolve Spec Version

     Build spec_id from spec number + version, and report file/folder presence.

    Args:
        spec_number (str): Base spec number, e.g. 38901
        version (None | str | Unset): Full version string, e.g. 19.1.0 or 'latest'
        major (int | None | Unset): Major version (0-35)
        minor (int | None | Unset): Minor version (0-9)
        patch (int | None | Unset): Patch version (0-9)
        docs_dir (None | str | Unset): Optional override for the specs directory. Defaults to
            specs_dir from spec_config.json.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SpecVersionResolveGetResponseSpecVersionResolveGet
    """

    return (
        await asyncio_detailed(
            client=client,
            spec_number=spec_number,
            version=version,
            major=major,
            minor=minor,
            patch=patch,
            docs_dir=docs_dir,
        )
    ).parsed
