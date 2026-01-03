from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.markdown_payload import MarkdownPayload
    from ..models.source_info import SourceInfo


T = TypeVar("T", bound="TableResponse")


@_attrs_define
class TableResponse:
    """
    Attributes:
        status (str):
        spec_id (str):
        table_id (str):
        caption (str):
        markdown (MarkdownPayload):
        source (SourceInfo):
        html (str):
    """

    status: str
    spec_id: str
    table_id: str
    caption: str
    markdown: MarkdownPayload
    source: SourceInfo
    html: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        spec_id = self.spec_id

        table_id = self.table_id

        caption = self.caption

        markdown = self.markdown.to_dict()

        source = self.source.to_dict()

        html = self.html

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "spec_id": spec_id,
                "table_id": table_id,
                "caption": caption,
                "markdown": markdown,
                "source": source,
                "html": html,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.markdown_payload import MarkdownPayload
        from ..models.source_info import SourceInfo

        d = dict(src_dict)
        status = d.pop("status")

        spec_id = d.pop("spec_id")

        table_id = d.pop("table_id")

        caption = d.pop("caption")

        markdown = MarkdownPayload.from_dict(d.pop("markdown"))

        source = SourceInfo.from_dict(d.pop("source"))

        html = d.pop("html")

        table_response = cls(
            status=status,
            spec_id=spec_id,
            table_id=table_id,
            caption=caption,
            markdown=markdown,
            source=source,
            html=html,
        )

        table_response.additional_properties = d
        return table_response

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
