from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.version_exists import VersionExists
    from ..models.version_paths import VersionPaths


T = TypeVar("T", bound="VersionResolveResponse")


@_attrs_define
class VersionResolveResponse:
    """
    Attributes:
        status (str):
        spec_number (str):
        version (str):
        spec_id (str):
        paths (VersionPaths):
        exists (VersionExists):
    """

    status: str
    spec_number: str
    version: str
    spec_id: str
    paths: VersionPaths
    exists: VersionExists
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        spec_number = self.spec_number

        version = self.version

        spec_id = self.spec_id

        paths = self.paths.to_dict()

        exists = self.exists.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "spec_number": spec_number,
                "version": version,
                "spec_id": spec_id,
                "paths": paths,
                "exists": exists,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.version_exists import VersionExists
        from ..models.version_paths import VersionPaths

        d = dict(src_dict)
        status = d.pop("status")

        spec_number = d.pop("spec_number")

        version = d.pop("version")

        spec_id = d.pop("spec_id")

        paths = VersionPaths.from_dict(d.pop("paths"))

        exists = VersionExists.from_dict(d.pop("exists"))

        version_resolve_response = cls(
            status=status,
            spec_number=spec_number,
            version=version,
            spec_id=spec_id,
            paths=paths,
            exists=exists,
        )

        version_resolve_response.additional_properties = d
        return version_resolve_response

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
