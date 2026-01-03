from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="GrepMatch")


@_attrs_define
class GrepMatch:
    """
    Attributes:
        index (int):
        line (int):
        char_offset (int):
        message_length (int):
        message (str):
        clause_id (None | str | Unset):
    """

    index: int
    line: int
    char_offset: int
    message_length: int
    message: str
    clause_id: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        index = self.index

        line = self.line

        char_offset = self.char_offset

        message_length = self.message_length

        message = self.message

        clause_id: None | str | Unset
        if isinstance(self.clause_id, Unset):
            clause_id = UNSET
        else:
            clause_id = self.clause_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "index": index,
                "line": line,
                "char_offset": char_offset,
                "message_length": message_length,
                "message": message,
            }
        )
        if clause_id is not UNSET:
            field_dict["clause_id"] = clause_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        index = d.pop("index")

        line = d.pop("line")

        char_offset = d.pop("char_offset")

        message_length = d.pop("message_length")

        message = d.pop("message")

        def _parse_clause_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        clause_id = _parse_clause_id(d.pop("clause_id", UNSET))

        grep_match = cls(
            index=index,
            line=line,
            char_offset=char_offset,
            message_length=message_length,
            message=message,
            clause_id=clause_id,
        )

        grep_match.additional_properties = d
        return grep_match

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
