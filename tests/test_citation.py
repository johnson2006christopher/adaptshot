"""`CITATION.cff` must stay valid and current (#24).

A citation file is metadata that only ever gets read by other people, so nothing
in normal development exercises it. That makes it exactly the kind of file that
drifts: the version stays at whatever it was when someone last thought about it,
and a stale citation is worse than none — it looks authoritative while pointing
at software that no longer exists.

`tests/test_release_metadata.py` already derives the expected version from
`pyproject.toml` for `src/adaptshot/__init__.py`. This applies the same rule to
the third place the version now lives.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CITATION_PATH = REPO_ROOT / "CITATION.cff"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

#: The minimum GitHub needs to render the "Cite this repository" button.
REQUIRED_KEYS = ("cff-version", "message", "title", "authors", "version", "date-released")


@pytest.fixture(scope="module")
def citation() -> dict[str, object]:
    yaml = pytest.importorskip("yaml", reason="PyYAML is only installed with an extra")
    loaded = yaml.safe_load(CITATION_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), "CITATION.cff must be a mapping"
    return loaded


def test_citation_file_exists() -> None:
    assert CITATION_PATH.is_file(), (
        "CITATION.cff is missing; GitHub's cite button disappears with it"
    )


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_required_keys_are_present(citation: dict[str, object], key: str) -> None:
    assert citation.get(key), f"CITATION.cff has no {key}"


def test_version_matches_pyproject(citation: dict[str, object]) -> None:
    """Three places now carry the version. They must agree.

    Read with a regex rather than `tomllib`, which is 3.11+ while this project
    supports 3.10 — the same reason `test_release_metadata.py` does.
    """

    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None, 'no `version = "..."` found in pyproject.toml'

    assert str(citation["version"]) == match.group(1), (
        f"CITATION.cff says {citation['version']}, pyproject.toml says "
        f"{match.group(1)}. Bump both, or the citation points at a release "
        "that is not the current one."
    )


def test_the_authors_are_named_not_placeheld(citation: dict[str, object]) -> None:
    """A citation with a placeholder author is not a citation."""

    authors = citation["authors"]
    assert isinstance(authors, list) and authors, "CITATION.cff lists no authors"
    for author in authors:
        assert isinstance(author, dict)
        assert author.get("family-names"), "an author has no family name"
        assert author.get("given-names"), "an author has no given names"


def test_no_placeholder_orcid_was_committed(citation: dict[str, object]) -> None:
    """The zero ORCID is the example from the spec, and identifies nobody.

    It is left out of the file deliberately, with a TODO, rather than filled in
    with something that looks real to a reader and resolves to nothing.
    """

    authors = citation["authors"]
    assert isinstance(authors, list)
    for author in authors:
        assert isinstance(author, dict)
        orcid = str(author.get("orcid", ""))
        assert "0000-0000-0000-0000" not in orcid, (
            "placeholder ORCID in CITATION.cff; remove it or register a real one"
        )


def test_the_readme_carries_a_copyable_citation() -> None:
    """People cite what they can paste on the day, not what they can find later."""

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "@software{" in readme or "@misc{" in readme, (
        "README has no BibTeX block; add one so the citation can be copied "
        "without leaving the page"
    )


def test_the_readme_and_citation_file_agree_on_the_title(
    citation: dict[str, object],
) -> None:
    """Two citation forms that disagree produce two entries for one project.

    The README block previously read "Few-Shot Vision with Calibrated,
    Guaranteed Uncertainty" while CITATION.cff read something else. Beyond the
    inconsistency, "Guaranteed" was an unearned word: conformal coverage has
    never been empirically validated (#14), and a bibliography entry is the
    least revisable place a claim can end up.
    """

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    title = str(citation["title"])
    subtitle = title.split(":", 1)[1].strip() if ":" in title else title

    # BibTeX wraps long titles across lines, so compare on collapsed whitespace.
    collapsed_readme = " ".join(readme.split()).lower()
    collapsed_subtitle = " ".join(subtitle.split()).lower()

    assert collapsed_subtitle in collapsed_readme, (
        "the README BibTeX title does not match CITATION.cff; one project "
        "should not have two names in the literature"
    )


def test_no_unvalidated_guarantee_in_the_citation(citation: dict[str, object]) -> None:
    """Coverage guarantees are #14's territory until measured (#17's lesson)."""

    title = str(citation["title"]).lower()
    assert "guarantee" not in title, (
        "the citation title claims a guarantee that has not been empirically "
        "validated; see #14"
    )


def test_the_dois_are_real_and_agree(citation: dict[str, object]) -> None:
    """The version DOI in CITATION.cff is the one in the README's BibTeX; the concept
    DOI in the badge is the one CITATION.cff lists as an identifier (#24)."""

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    version_doi = str(citation["doi"])
    assert re.fullmatch(r"10\.5281/zenodo\.\d+", version_doi), f"not a Zenodo DOI: {version_doi}"
    assert f"doi     = {{{version_doi}}}" in readme, "README BibTeX does not carry the version DOI"

    identifiers = citation.get("identifiers")
    assert isinstance(identifiers, list) and identifiers, "CITATION.cff lists no concept DOI"
    concept = [i for i in identifiers if isinstance(i, dict) and i.get("type") == "doi"]
    assert concept, "no DOI-typed identifier for the concept DOI"
    concept_doi = str(concept[0]["value"])
    assert concept_doi != version_doi, "concept DOI must differ from the version DOI"
    badge = f"[![DOI](https://zenodo.org/badge/DOI/{concept_doi}.svg)](https://doi.org/{concept_doi})"
    assert badge in readme, "README badge does not carry the concept DOI"
    assert "0000-0000" not in version_doi + concept_doi
