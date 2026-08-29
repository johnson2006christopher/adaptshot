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


def test_torch_free_inference_is_backed_by_bundled_weights() -> None:
    """The claim is now true, so this guards it from becoming false again.

    It used to assert the opposite: `src/adaptshot/data/` was empty, so nothing
    was allowed to advertise torch-free inference, and the test was written to
    retire when #36 landed. #36 has landed. Rather than deleting it, it is
    inverted -- the failure mode has flipped from "claiming something untrue" to
    "the weights that make it true went missing", and that is worth catching.

    A wheel built without the ONNX graph would install cleanly and then fail on
    the first `predict()` of anyone who did not also install torch.
    """

    bundled = sorted((REPO_ROOT / "src" / "adaptshot" / "data").glob("*.onnx"))
    assert bundled, (
        "no ONNX backbone is bundled, but the documentation says inference "
        "works without torch. Either run scripts/export_backbones.py, or the "
        "claim has to come back out of the docs (#36)."
    )

    default = REPO_ROOT / "src" / "adaptshot" / "data" / "mobilenet_v3_small.onnx"
    assert default.is_file(), (
        "the default backbone's ONNX graph is missing; a core install would "
        "fail on its first prediction"
    )

    weights = default.with_suffix(".onnx.data")
    assert weights.is_file(), (
        "the ONNX graph is present but its external weights file is not. The "
        "graph alone is 0.3MB and loads without error, then produces nothing."
    )


PLANTVILLAGE_RESULT_PATH = REPO_ROOT / "results" / "plantvillage_5way5shot.json"


@pytest.mark.parametrize(
    ("label", "path", "fmt"),
    [
        ("embedding, per image", ("embedding_ms", "median"), "{:.1f} ms"),
        ("support fit, per episode", ("support_fit_ms", "median"), "{:.0f} ms"),
        ("predict, per query", ("predict_ms", "median"), "{:.1f} ms"),
        ("cold start", ("cold_start", "seconds"), "{:.2f} s"),
    ],
)
def test_readme_latency_table_matches_the_plantvillage_artifact(
    label: str, path: tuple[str, str], fmt: str
) -> None:
    """Every performance figure in the README traces to the run that produced it (#20).

    The README's table is written *from* the artifact, so this is not a check
    that two hand-typed numbers agree; it is a check that nobody edits one and
    forgets the other, in either direction.
    """

    record = json.loads(PLANTVILLAGE_RESULT_PATH.read_text(encoding="utf-8"))["timing"]
    expected = fmt.format(record[path[0]][path[1]])
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    row = next((line for line in readme.splitlines() if line.startswith(f"| {label} |")), None)
    assert row is not None, f"README has no latency row for {label!r}"
    assert f"**{expected}**" in row, (
        f"README row for {label!r} does not quote {expected} from the artifact:\n  {row}"
    )


def test_readme_memory_figure_is_the_single_cycle_not_the_harness() -> None:
    """The 120 MB claim is one process, one support set, one answer -- and it must be
    that number, not the benchmark process's, which holds far more."""

    record = json.loads(PLANTVILLAGE_RESULT_PATH.read_text(encoding="utf-8"))["timing"]
    cycle = record["cold_start"]["peak_rss_mb"]
    harness = record["benchmark_process_peak_rss_mb"]
    assert harness > cycle * 2, "the harness should cost far more than one cycle; if not, the split is wrong"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"**Peak memory for that cold-start cycle: {cycle:.0f} MB**" in readme
    assert f"harness itself peaks at {harness:.0f} MB" in readme



def test_technical_note_figures_trace_to_the_artifact() -> None:
    """The note's numbers are formatted from the artifact here and must appear verbatim (#26).

    Two pages of claims are exactly the place a stale figure hides longest, so
    every headline number in docs/technical-note.md is checked against the
    file it says it comes from -- accuracy, coverage, set size, the baseline it
    is compared to, latency by stage, cold start, and the single-cycle memory.
    """

    record = json.loads(PLANTVILLAGE_RESULT_PATH.read_text(encoding="utf-8"))
    acc = record["accuracy"]
    conformal = record["conformal"]
    top1 = record["top1_threshold"]
    timing = record["timing"]

    expected = [
        f"{acc['adaptshot']['mean'] * 100:.1f}% ± {acc['adaptshot']['ci95_half_width'] * 100:.1f}",
        f"{acc['nearest_centroid']['mean'] * 100:.1f}% ± {acc['nearest_centroid']['ci95_half_width'] * 100:.1f}",
        f"{acc['linear_probe']['mean'] * 100:.1f}% ± {acc['linear_probe']['ci95_half_width'] * 100:.1f}",
        f"{acc['knn_1']['mean'] * 100:.1f}% ± {acc['knn_1']['ci95_half_width'] * 100:.1f}",
        f"{acc['knn_5']['mean'] * 100:.1f}% ± {acc['knn_5']['ci95_half_width'] * 100:.1f}",
        f"{conformal['empirical_coverage']['mean'] * 100:.1f}% ± {conformal['empirical_coverage']['ci95_half_width'] * 100:.1f}",
        f"{conformal['mean_set_size']['mean']:.2f} ± {conformal['mean_set_size']['ci95_half_width']:.2f}",
        f"{top1['coverage']['mean'] * 100:.1f}% ± {top1['coverage']['ci95_half_width'] * 100:.1f}",
        f"{timing['embedding_ms']['median']:.1f} ms | {timing['embedding_ms']['p95']:.1f} ms",
        f"{timing['support_fit_ms']['median']:.0f} ms | {timing['support_fit_ms']['p95']:.0f} ms",
        f"{timing['predict_ms']['median']:.1f} ms | {timing['predict_ms']['p95']:.1f} ms",
        f"{timing['cold_start']['seconds']:.2f} s",
        f"**{timing['cold_start']['peak_rss_mb']:.0f} MB**",
        f"seed {record['protocol']['seed']}",
        f"{record['protocol']['episodes']} episodes",
    ]
    note = (REPO_ROOT / "docs" / "technical-note.md").read_text(encoding="utf-8")
    missing = [text for text in expected if text not in note]
    assert not missing, (
        "docs/technical-note.md does not quote these figures from "
        "results/plantvillage_5way5shot.json:\n  " + "\n  ".join(missing)
    )


def test_readme_results_tables_trace_to_the_artifact() -> None:
    """The accuracy and conformal tables, under the same guard as the latency table.

    The linear probe's interval was quoted as 1.2 in three places; the artifact
    says 1.1466, which is 1.1. That was rounded by eye from a two-decimal
    printout and nothing checked it. Now something does.
    """

    record = json.loads(PLANTVILLAGE_RESULT_PATH.read_text(encoding="utf-8"))
    acc = record["accuracy"]
    conformal = record["conformal"]
    top1 = record["top1_threshold"]

    def pct(block: dict[str, float]) -> str:
        return f"{block['mean'] * 100:.1f}% ± {block['ci95_half_width'] * 100:.1f}"

    def num(block: dict[str, float]) -> str:
        return f"{block['mean']:.2f} ± {block['ci95_half_width']:.2f}"

    expected = {
        "AdaptShot accuracy": pct(acc["adaptshot"]),
        "nearest centroid": pct(acc["nearest_centroid"]),
        "linear probe": pct(acc["linear_probe"]),
        "1-NN": pct(acc["knn_1"]),
        "5-NN": pct(acc["knn_5"]),
        "conformal coverage": pct(conformal["empirical_coverage"]),
        "conformal set size": num(conformal["mean_set_size"]),
        "threshold coverage": pct(top1["coverage"]),
        "threshold set size": num(top1["mean_set_size"]),
    }
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    missing = [f"{label}: {text}" for label, text in expected.items() if text not in readme]
    assert not missing, "README results tables disagree with the artifact:\n  " + "\n  ".join(missing)

