"""Guard the documentation against claims the repository does not support (#17).

Documentation drifts from code silently. Nothing fails, nothing goes red -- a sentence
simply stops being true and stays on the site. This repository shipped four such
sentences at once: an application described as deployed when it had never run outside a
test, "torch-free inference via bundled backbones" when ``src/adaptshot/data/`` held
only ``__init__.py``, a latency figure of 20ms citing a benchmark whose own artifact
recorded 36ms, and links to a filesystem path on a machine that no longer exists.

The remedy for a number is not to check it once. It is to make the document and the
artifact the same source, so that a stale claim fails a test instead of misleading a
reader.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
SMOKE_RESULT_PATH = REPO_ROOT / "results" / "smoke_test.json"

# Historical records, deliberately preserved as written. They carry a banner saying so.
_HISTORICAL = {"archive/audit-report-v0.1.1.md", "release-checklist-v0.1.1.md"}


def _live_docs() -> list[Path]:
    """Documentation that speaks in the present tense about the current release."""

    return [
        path
        for path in sorted(DOCS_ROOT.rglob("*.md"))
        if path.relative_to(DOCS_ROOT).as_posix() not in _HISTORICAL
    ]


def _smoke_result() -> dict[str, object]:
    return json.loads(SMOKE_RESULT_PATH.read_text(encoding="utf-8"))


def test_no_absolute_filesystem_links() -> None:
    """`file:///` links leak a local path and are broken for every reader."""

    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{n}"
        for path in DOCS_ROOT.rglob("*.md")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "file:///" in line
    ]
    assert not offenders, (
        "absolute filesystem links in documentation:\n  " + "\n  ".join(offenders)
    )


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """Blank-line-separated blocks with the line each starts on.

    Line-by-line matching is the wrong unit for prose: markdown wraps sentences at
    arbitrary columns, so a claim and the word negating it routinely land on different
    lines. Checking paragraphs is what a reader actually experiences.
    """

    blocks: list[tuple[int, str]] = []
    start, buffer = 1, []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            if not buffer:
                start = n
            buffer.append(line)
        elif buffer:
            blocks.append((start, " ".join(buffer)))
            buffer = []
    if buffer:
        blocks.append((start, " ".join(buffer)))
    return blocks


def test_mziziguard_is_never_described_as_deployed() -> None:
    """MziziGuard has never been deployed; no live document may say otherwise."""

    banned = re.compile(r"\b(deployed|production-ready)\b", re.IGNORECASE)
    negation = re.compile(r"\b(not|never|neither|no longer|has not|was not)\b", re.IGNORECASE)
    offenders = []
    for path in _live_docs():
        for line_no, block in _paragraphs(path.read_text(encoding="utf-8")):
            if "mzizi" not in block.lower() or not banned.search(block):
                continue
            # A paragraph saying it is *not* deployed is the correction, not the claim.
            if negation.search(block):
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {block[:120]}")
    assert not offenders, (
        "MziziGuard described as deployed or production-ready:\n  "
        + "\n  ".join(offenders)
        + "\n\nIt has never been deployed. See #17."
    )


def test_documents_mentioning_mziziguard_disclose_synthetic_data() -> None:
    """The synthetic-data disclosure must travel with the claim, not live one page away.

    A reader arrives on one page, not on the site. If a document presents MziziGuard's
    results, that document must say the images are generated.
    """

    required = re.compile(r"synthetic|generated|procedural|drawn|illustrat", re.IGNORECASE)
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _live_docs()
        if "mzizi" in path.read_text(encoding="utf-8").lower()
        and not required.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "documents present MziziGuard without disclosing that its data is synthetic:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    ("field", "pattern", "scale"),
    [
        ("accuracy", r"(\d+)%\s+(?:accuracy\s+)?on the (?:5-way 10-shot )?CIFAR-10 smoke", 100.0),
    ],
)
def test_quoted_benchmark_figures_match_the_artifact(
    field: str, pattern: str, scale: float
) -> None:
    """A number quoted in the docs must equal the artifact it claims to come from.

    This is the check that would have caught "20ms P95 latency" the day it was written:
    the benchmark it cited recorded 36ms in the very file the docs point at.
    """

    actual = float(_smoke_result()[field])  # type: ignore[arg-type]
    quoted = [
        (f"{path.relative_to(REPO_ROOT)}", int(match.group(1)))
        for path in _live_docs()
        for match in re.finditer(pattern, path.read_text(encoding="utf-8"))
    ]
    assert quoted, f"no document quotes the benchmark {field}; the guard is not watching anything"

    mismatched = [
        f"{where} claims {value}% but results/smoke_test.json records {actual * scale:.0f}%"
        for where, value in quoted
        if abs(value - actual * scale) > 0.5
    ]
    assert not mismatched, "docs disagree with the benchmark artifact:\n  " + "\n  ".join(mismatched)


def test_torch_free_inference_is_not_advertised_as_working() -> None:
    """`src/adaptshot/data/` ships no backbones, so nothing may claim torch-free inference.

    When #36 lands and real ONNX weights are bundled, this test starts failing and
    should be deleted -- that failure is the signal the claim became true.
    """

    bundled = list((REPO_ROOT / "src" / "adaptshot" / "data").glob("*.onnx"))
    if bundled:
        pytest.skip(f"{len(bundled)} bundled backbone(s) present; see #36 and remove this test")

    claim = re.compile(r"torch-free[^.\n]*\b(via|using|through)\b[^.\n]*bundled", re.IGNORECASE)
    offenders = []
    for path in _live_docs():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = claim.search(line)
            if match is None:
                continue
            # A phrase inside quotation marks is being cited, not asserted. The
            # changelog quotes this claim in order to retract it.
            if re.search(r'["\u201c\u201d]', line[: match.start()]):
                continue
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{n}")
    assert not offenders, (
        "torch-free inference via bundled backbones is advertised, but "
        "src/adaptshot/data/ contains no .onnx files:\n  " + "\n  ".join(offenders)
    )
