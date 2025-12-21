"""CLI tools for spec-document TOC inspection and HTML section extraction."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .spec_docs import (
    SpecDocError,
    extract_section_html,
    filter_toc_entries,
    format_toc_as_text,
    html_fragment_to_text,
    html_fragment_to_markdown,
    load_toc_entries,
    load_toc_json,
    resolve_doc_paths,
    resolve_section_html_id,
)


@click.command("asntools-spec-toc")
@click.argument("doc")
@click.option(
    "--docs-dir",
    type=click.Path(path_type=Path),
    default=None,
    show_default=False,
    help="Override specs directory (defaults to specs_dir in spec_config.json or ./doc)",
)
@click.option("--prefix", help="Filter by clause/id prefix, e.g. 4.7 or 4-7")
@click.option("--search", help="Case-insensitive substring search")
@click.option("--max-depth", type=int, help="Limit output to a maximum depth from root")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text", show_default=True)
def toc(doc: str, docs_dir: Path, prefix: str | None, search: str | None, max_depth: int | None, output_format: str) -> None:
    """Print the table of contents for DOC (basename or path)."""

    _, toc_path = resolve_doc_paths(doc, docs_dir=docs_dir)
    try:
        toc_json = load_toc_json(toc_path)
        entries = load_toc_entries(toc_json)
        items = filter_toc_entries(entries, prefix=prefix, search=search, max_depth=max_depth)
    except SpecDocError as err:
        raise click.ClickException(str(err)) from err

    if output_format == "json":
        payload = [
            {
                "depth": depth,
                "clause_id": entry.clause_id,
                "clause_title": entry.clause_title,
                "level": entry.level,
                "id": entry.html_id,
            }
            for depth, entry in items
        ]
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    click.echo(format_toc_as_text(items))


@click.command("asntools-spec-extract")
@click.argument("doc")
@click.argument("section")
@click.option(
    "--docs-dir",
    type=click.Path(path_type=Path),
    default=None,
    show_default=False,
    help="Override specs directory (defaults to specs_dir in spec_config.json or ./doc)",
)
@click.option("--format", "output_format", type=click.Choice(["md", "text", "html"]), default="md", show_default=True)
@click.option("--include-heading/--no-include-heading", default=True, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), help="Write output to a specific file path")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("artifacts/spec_sections"),
    show_default=True,
    help="Directory for auto-named markdown output when --format md and --output is not set",
)
def extract(
    doc: str,
    section: str,
    docs_dir: Path,
    output_format: str,
    include_heading: bool,
    output: Path | None,
    out_dir: Path,
) -> None:
    """Extract a section from DOC by clause id (e.g. 4.7.1) or heading id/prefix (e.g. 4-7-1)."""

    html_path, toc_path = resolve_doc_paths(doc, docs_dir=docs_dir)
    try:
        toc_json = load_toc_json(toc_path)
        entries = load_toc_entries(toc_json)
        section_html_id = resolve_section_html_id(entries, section)
        fragment = extract_section_html(html_path, section_html_id, include_heading=include_heading)
    except SpecDocError as err:
        raise click.ClickException(str(err)) from err

    if output_format == "html":
        rendered = fragment
    elif output_format == "text":
        rendered = html_fragment_to_text(fragment)
    else:
        rendered = html_fragment_to_markdown(fragment)

    if output_format == "md":
        if output is None:
            safe_section = section_html_id.replace("/", "-")
            output = out_dir / f"{html_path.stem}__{safe_section}.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        click.echo(str(output))
        return

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        click.echo(f"Wrote {output}")
        return

    click.echo(rendered)


if __name__ == "__main__":  # pragma: no cover
    toc()
