"""Files that repeat each other stay in step: CITATION.cff with the package version and
the citing page, PROVIDERS.md and MIGRATION.md with their docs pages, and the example
notebooks stay free of outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "website" / "src" / "content" / "docs"


def test_citation_matches_the_package_and_the_citing_page() -> None:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation = yaml.safe_load(text)
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    assert match is not None
    version = match.group(1)
    assert citation["cff-version"] == "1.2.0"
    assert str(citation["version"]) == version, "bump CITATION.cff with the release"
    for key in ("title", "authors", "message", "license", "repository-code", "doi"):
        assert citation.get(key), key
    page = (DOCS / "project" / "citing.mdx").read_text(encoding="utf-8")
    block = re.search(r'<TabItem label="CFF">\s*```yaml\n(.*?)\n\s*```', page, re.S)
    assert block is not None
    shown = "\n".join(
        line[4:] if line.startswith("    ") else line for line in block.group(1).splitlines()
    )
    assert shown.strip() == text.strip(), "the citing page's CFF tab must equal CITATION.cff"
    assert f"version = {{{version}}}" in page


def _blocks(markdown: str) -> List[str]:
    """Paragraphs, list items and table rows; headings are left out (pages restructure them)."""
    blocks: List[str] = []
    for paragraph in re.split(r"\n\s*\n", markdown):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("#"):
            continue
        if re.match(r"([-*|]|\d+\.)\s", paragraph) or paragraph.startswith("|"):
            blocks.extend(line.strip() for line in paragraph.splitlines() if line.strip())
        else:
            blocks.append(paragraph)
    return blocks


def _normal(text: str) -> str:
    text = text.replace("https://tahamukhtar20.github.io/mapcv/", "/mapcv/")
    return " ".join(text.split())


@pytest.mark.parametrize(
    ("source", "page"),
    [("PROVIDERS.md", "project/providers.mdx"), ("MIGRATION.md", "project/migration.mdx")],
)
def test_docs_pages_carry_every_paragraph_of_their_source(source: str, page: str) -> None:
    rendered = _normal((DOCS / page).read_text(encoding="utf-8"))
    blocks = _blocks((ROOT / source).read_text(encoding="utf-8"))
    missing = [block for block in blocks if _normal(block) not in rendered]
    assert len(blocks) > 5
    assert not missing, (
        f"{page} lacks text from {source} (the source is authoritative):\n" + "\n".join(missing)
    )


@pytest.mark.parametrize(
    "notebook", sorted((ROOT / "examples").rglob("*.ipynb")), ids=lambda p: p.name
)
def test_example_notebooks_are_stored_without_outputs(notebook: Path) -> None:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code
    assert all(not cell.get("outputs") and cell.get("execution_count") is None for cell in code), (
        f"clear the outputs of {notebook.name} before committing (Kernel > Restart & Clear Output)"
    )
