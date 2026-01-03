from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.image_payload import ImagePayload


T = TypeVar("T", bound="MarkdownChunk")


@_attrs_define
class MarkdownChunk:
    """
    Attributes:
        index (int):
        bytes_ (int):
        md_snippet (str):
        images (list[ImagePayload] | Unset):
    """

    index: int
    bytes_: int
    md_snippet: str
    images: list[ImagePayload] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        index = self.index

        bytes_ = self.bytes_

        md_snippet = self.md_snippet

        images: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.images, Unset):
            images = []
            for images_item_data in self.images:
                images_item = images_item_data.to_dict()
                images.append(images_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "index": index,
                "bytes": bytes_,
                "md_snippet": md_snippet,
            }
        )
        if images is not UNSET:
            field_dict["images"] = images

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.image_payload import ImagePayload

        d = dict(src_dict)
        index = d.pop("index")

        bytes_ = d.pop("bytes")

        md_snippet = d.pop("md_snippet")

        _images = d.pop("images", UNSET)
        images: list[ImagePayload] | Unset = UNSET
        if _images is not UNSET:
            images = []
            for images_item_data in _images:
                images_item = ImagePayload.from_dict(images_item_data)

                images.append(images_item)

        markdown_chunk = cls(
            index=index,
            bytes_=bytes_,
            md_snippet=md_snippet,
            images=images,
        )

        markdown_chunk.additional_properties = d
        return markdown_chunk

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
