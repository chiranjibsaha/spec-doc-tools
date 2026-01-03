from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.grep_match import GrepMatch
    from ..models.source_info import SourceInfo


T = TypeVar("T", bound="GrepResult")


@_attrs_define
class GrepResult:
    """
    Attributes:
        status (str):
        spec_id (str):
        query (str):
        match_count (int):
        matches (list[GrepMatch]):
        chunks (list[str]):
        source (SourceInfo):
    """

    status: str
    spec_id: str
    query: str
    match_count: int
    matches: list[GrepMatch]
    chunks: list[str]
    source: SourceInfo
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        spec_id = self.spec_id

        query = self.query

        match_count = self.match_count

        matches = []
        for matches_item_data in self.matches:
            matches_item = matches_item_data.to_dict()
            matches.append(matches_item)

        chunks = self.chunks

        source = self.source.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "spec_id": spec_id,
                "query": query,
                "match_count": match_count,
                "matches": matches,
                "chunks": chunks,
                "source": source,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.grep_match import GrepMatch
        from ..models.source_info import SourceInfo

        d = dict(src_dict)
        status = d.pop("status")

        spec_id = d.pop("spec_id")

        query = d.pop("query")

        match_count = d.pop("match_count")

        matches = []
        _matches = d.pop("matches")
        for matches_item_data in _matches:
            matches_item = GrepMatch.from_dict(matches_item_data)

            matches.append(matches_item)

        chunks = cast(list[str], d.pop("chunks"))

        source = SourceInfo.from_dict(d.pop("source"))

        grep_result = cls(
            status=status,
            spec_id=spec_id,
            query=query,
            match_count=match_count,
            matches=matches,
            chunks=chunks,
            source=source,
        )

        grep_result.additional_properties = d
        return grep_result

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
