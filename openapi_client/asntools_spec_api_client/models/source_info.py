from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SourceInfo")


@_attrs_define
class SourceInfo:
    """
    Attributes:
        html_path (None | str | Unset):
        toc_path (None | str | Unset):
    """

    html_path: None | str | Unset = UNSET
    toc_path: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        html_path: None | str | Unset
        if isinstance(self.html_path, Unset):
            html_path = UNSET
        else:
            html_path = self.html_path

        toc_path: None | str | Unset
        if isinstance(self.toc_path, Unset):
            toc_path = UNSET
        else:
            toc_path = self.toc_path

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if html_path is not UNSET:
            field_dict["html_path"] = html_path
        if toc_path is not UNSET:
            field_dict["toc_path"] = toc_path

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_html_path(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        html_path = _parse_html_path(d.pop("html_path", UNSET))

        def _parse_toc_path(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        toc_path = _parse_toc_path(d.pop("toc_path", UNSET))

        source_info = cls(
            html_path=html_path,
            toc_path=toc_path,
        )

        source_info.additional_properties = d
        return source_info

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
