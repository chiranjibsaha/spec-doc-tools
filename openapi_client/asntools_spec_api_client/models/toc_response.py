from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field


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
    """

    status: str
    spec_id: str
    depth_limit: int | None
    toc: list[TOCItem]
    source: SourceInfo
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

        toc_response = cls(
            status=status,
            spec_id=spec_id,
            depth_limit=depth_limit,
            toc=toc,
            source=source,
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
