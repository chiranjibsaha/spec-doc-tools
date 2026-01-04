from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TOCItem")


@_attrs_define
class TOCItem:
    """
    Attributes:
        depth (int):
        clause_id (None | str):
        clause_title (str):
        level (int):
        clause_id_ref (str):
    """

    depth: int
    clause_id: None | str
    clause_title: str
    level: int
    clause_id_ref: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        depth = self.depth

        clause_id: None | str
        clause_id = self.clause_id

        clause_title = self.clause_title

        level = self.level

        clause_id_ref = self.clause_id_ref

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "depth": depth,
                "clause_id": clause_id,
                "clause_title": clause_title,
                "level": level,
                "clause_id_ref": clause_id_ref,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        depth = d.pop("depth")

        def _parse_clause_id(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        clause_id = _parse_clause_id(d.pop("clause_id"))

        clause_title = d.pop("clause_title")

        level = d.pop("level")

        clause_id_ref = d.pop("clause_id_ref")

        toc_item = cls(
            depth=depth,
            clause_id=clause_id,
            clause_title=clause_title,
            level=level,
            clause_id_ref=clause_id_ref,
        )

        toc_item.additional_properties = d
        return toc_item

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
