from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.source_info import SourceInfo


T = TypeVar("T", bound="SectionSummaryResponse")


@_attrs_define
class SectionSummaryResponse:
    """
    Attributes:
        status (str):
        spec_id (str):
        section_ref (str):
        html_id (str):
        include_heading (bool):
        chars (int):
        bytes_ (int):
        source (SourceInfo):
    """

    status: str
    spec_id: str
    section_ref: str
    html_id: str
    include_heading: bool
    chars: int
    bytes_: int
    source: SourceInfo
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        spec_id = self.spec_id

        section_ref = self.section_ref

        html_id = self.html_id

        include_heading = self.include_heading

        chars = self.chars

        bytes_ = self.bytes_

        source = self.source.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "spec_id": spec_id,
                "section_ref": section_ref,
                "html_id": html_id,
                "include_heading": include_heading,
                "chars": chars,
                "bytes": bytes_,
                "source": source,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.source_info import SourceInfo

        d = dict(src_dict)
        status = d.pop("status")

        spec_id = d.pop("spec_id")

        section_ref = d.pop("section_ref")

        html_id = d.pop("html_id")

        include_heading = d.pop("include_heading")

        chars = d.pop("chars")

        bytes_ = d.pop("bytes")

        source = SourceInfo.from_dict(d.pop("source"))

        section_summary_response = cls(
            status=status,
            spec_id=spec_id,
            section_ref=section_ref,
            html_id=html_id,
            include_heading=include_heading,
            chars=chars,
            bytes_=bytes_,
            source=source,
        )

        section_summary_response.additional_properties = d
        return section_summary_response

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
