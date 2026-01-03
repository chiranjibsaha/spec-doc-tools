from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ImagePayload")


@_attrs_define
class ImagePayload:
    """
    Attributes:
        index (int):
        src (str):
        alt (str):
        path (str):
        content_type (str):
        found (bool):
        bytes_ (int | None | Unset):
        base64 (None | str | Unset):
        svg (None | str | Unset):
    """

    index: int
    src: str
    alt: str
    path: str
    content_type: str
    found: bool
    bytes_: int | None | Unset = UNSET
    base64: None | str | Unset = UNSET
    svg: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        index = self.index

        src = self.src

        alt = self.alt

        path = self.path

        content_type = self.content_type

        found = self.found

        bytes_: int | None | Unset
        if isinstance(self.bytes_, Unset):
            bytes_ = UNSET
        else:
            bytes_ = self.bytes_

        base64: None | str | Unset
        if isinstance(self.base64, Unset):
            base64 = UNSET
        else:
            base64 = self.base64

        svg: None | str | Unset
        if isinstance(self.svg, Unset):
            svg = UNSET
        else:
            svg = self.svg

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "index": index,
                "src": src,
                "alt": alt,
                "path": path,
                "content_type": content_type,
                "found": found,
            }
        )
        if bytes_ is not UNSET:
            field_dict["bytes"] = bytes_
        if base64 is not UNSET:
            field_dict["base64"] = base64
        if svg is not UNSET:
            field_dict["svg"] = svg

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        index = d.pop("index")

        src = d.pop("src")

        alt = d.pop("alt")

        path = d.pop("path")

        content_type = d.pop("content_type")

        found = d.pop("found")

        def _parse_bytes_(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        bytes_ = _parse_bytes_(d.pop("bytes", UNSET))

        def _parse_base64(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        base64 = _parse_base64(d.pop("base64", UNSET))

        def _parse_svg(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        svg = _parse_svg(d.pop("svg", UNSET))

        image_payload = cls(
            index=index,
            src=src,
            alt=alt,
            path=path,
            content_type=content_type,
            found=found,
            bytes_=bytes_,
            base64=base64,
            svg=svg,
        )

        image_payload.additional_properties = d
        return image_payload

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
