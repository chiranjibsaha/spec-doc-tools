from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.source_info import SourceInfo
    from ..models.toc_item import TOCItem


T = TypeVar("T", bound="TOCResponse")


@_attrs_define
class TOCResponse:
    """
    Attributes:
        status (str):
        spec_id (str):
        depth_limit (int | None):
        toc (list[TOCItem]):
        source (SourceInfo):
        section_ref (None | str | Unset):
        html_id (None | str | Unset):
        section_text (None | str | Unset):
    """

    status: str
    spec_id: str
    depth_limit: int | None
    toc: list[TOCItem]
    source: SourceInfo
    section_ref: None | str | Unset = UNSET
    html_id: None | str | Unset = UNSET
    section_text: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        spec_id = self.spec_id

        depth_limit: int | None
        depth_limit = self.depth_limit

        toc = []
        for toc_item_data in self.toc:
            toc_item = toc_item_data.to_dict()
            toc.append(toc_item)

        source = self.source.to_dict()

        section_ref: None | str | Unset
        if isinstance(self.section_ref, Unset):
            section_ref = UNSET
        else:
            section_ref = self.section_ref

        html_id: None | str | Unset
        if isinstance(self.html_id, Unset):
            html_id = UNSET
        else:
            html_id = self.html_id

        section_text: None | str | Unset
        if isinstance(self.section_text, Unset):
            section_text = UNSET
        else:
            section_text = self.section_text

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "spec_id": spec_id,
                "depth_limit": depth_limit,
                "toc": toc,
                "source": source,
            }
        )
        if section_ref is not UNSET:
            field_dict["section_ref"] = section_ref
        if html_id is not UNSET:
            field_dict["html_id"] = html_id
        if section_text is not UNSET:
            field_dict["section_text"] = section_text

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.source_info import SourceInfo
        from ..models.toc_item import TOCItem

        d = dict(src_dict)
        status = d.pop("status")

        spec_id = d.pop("spec_id")

        def _parse_depth_limit(data: object) -> int | None:
            if data is None:
                return data
            return cast(int | None, data)

        depth_limit = _parse_depth_limit(d.pop("depth_limit"))

        toc = []
        _toc = d.pop("toc")
        for toc_item_data in _toc:
            toc_item = TOCItem.from_dict(toc_item_data)

            toc.append(toc_item)

        source = SourceInfo.from_dict(d.pop("source"))

        def _parse_section_ref(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        section_ref = _parse_section_ref(d.pop("section_ref", UNSET))

        def _parse_html_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        html_id = _parse_html_id(d.pop("html_id", UNSET))

        def _parse_section_text(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        section_text = _parse_section_text(d.pop("section_text", UNSET))

        toc_response = cls(
            status=status,
            spec_id=spec_id,
            depth_limit=depth_limit,
            toc=toc,
            source=source,
            section_ref=section_ref,
            html_id=html_id,
            section_text=section_text,
        )

        toc_response.additional_properties = d
        return toc_response

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
