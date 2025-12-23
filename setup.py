from __future__ import annotations

import pathlib

from setuptools import find_packages, setup

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = pathlib.Path(__file__).parent
PYPROJECT = ROOT / "pyproject.toml"


def load_project_metadata() -> dict:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    return {
        "name": project.get("name", "spec-doc-tools"),
        "version": project.get("version", "0.0.0"),
        "description": project.get("description", ""),
        "dependencies": project.get("dependencies", []),
        "optional_dependencies": project.get("optional-dependencies", {}),
        "readme": project.get("readme", "README.md"),
        "requires_python": project.get("requires-python", ">=3.10"),
    }


meta = load_project_metadata()
long_description = (ROOT / meta["readme"]).read_text(encoding="utf-8") if meta.get("readme") else ""

setup(
    name=meta["name"],
    version=meta["version"],
    description=meta["description"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=meta["requires_python"],
    packages=find_packages(include=["spec_doc_tools", "remote_mcp_client", "spec_doc_tools.*", "remote_mcp_client.*"]),
    install_requires=meta["dependencies"],
    extras_require=meta["optional_dependencies"],
)
