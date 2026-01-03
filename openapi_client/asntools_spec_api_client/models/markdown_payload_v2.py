from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.markdown_chunk import MarkdownChunk


T = TypeVar("T", bound="MarkdownPayloadV2")


@_attrs_define
class MarkdownPayloadV2:
    """
    Attributes:
        bytes_ (int):
        md (str):
        chunk_count (int):
        chunk_size (int):
        chunks (list[MarkdownChunk]):
    """

    bytes_: int
    md: str
    chunk_count: int
    chunk_size: int
    chunks: list[MarkdownChunk]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        bytes_ = self.bytes_

        md = self.md

        chunk_count = self.chunk_count

        chunk_size = self.chunk_size

        chunks = []
        for chunks_item_data in self.chunks:
            chunks_item = chunks_item_data.to_dict()
            chunks.append(chunks_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "bytes": bytes_,
                "md": md,
                "chunk_count": chunk_count,
                "chunk_size": chunk_size,
                "chunks": chunks,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.markdown_chunk import MarkdownChunk

        d = dict(src_dict)
        bytes_ = d.pop("bytes")

        md = d.pop("md")

        chunk_count = d.pop("chunk_count")

        chunk_size = d.pop("chunk_size")

        chunks = []
        _chunks = d.pop("chunks")
        for chunks_item_data in _chunks:
            chunks_item = MarkdownChunk.from_dict(chunks_item_data)

            chunks.append(chunks_item)

        markdown_payload_v2 = cls(
            bytes_=bytes_,
            md=md,
            chunk_count=chunk_count,
            chunk_size=chunk_size,
            chunks=chunks,
        )

        markdown_payload_v2.additional_properties = d
        return markdown_payload_v2

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
