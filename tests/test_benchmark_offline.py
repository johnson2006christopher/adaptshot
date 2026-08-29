"""The documented validation gate must run without a network (#12).

`--smoke-test` called `datasets.CIFAR10(download=True)` unconditionally. The
command contributors are told to run before opening a PR therefore required a
~170MB download -- from a library whose stated reason to exist is that
connectivity is the resource its users do not have.

It did not fail politely either. Measured on a GitHub runner, the download took
**34 minutes 48 seconds** while the benchmark itself took 56 seconds: 97% of the
job was waiting for data. On a slower link it produced silence and then a
timeout with no explanation.

The offline path is now the default. These tests keep it that way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "results" / "smoke_test.json"


@pytest.fixture(scope="module")
def benchmark():  # type: ignore[no-untyped-def]
    pytest.importorskip("torch", reason="the benchmark imports torch at module scope")
    from benchmarks import run_benchmark

    return run_benchmark


# ---------------------------------------------------------------------------
# Which dataset gets used
# ---------------------------------------------------------------------------


def test_auto_falls_back_to_the_offline_fixture(benchmark, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """No cache and no permission to download must not mean no benchmark."""

    assert benchmark.resolve_dataset("auto", str(tmp_path), allow_download=False) == "synthetic"


def test_auto_prefers_real_data_when_it_is_already_there(benchmark, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """A cached dataset costs nothing to use, so the real number stays the default."""

    marker = tmp_path / "cifar-10-batches-py" / "data_batch_1"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"")

    assert benchmark.resolve_dataset("auto", str(tmp_path), allow_download=False) == "cifar10"


def test_auto_uses_cifar_when_downloading_is_permitted(benchmark, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert benchmark.resolve_dataset("auto", str(tmp_path), allow_download=True) == "cifar10"


def test_an_explicit_dataset_is_never_overridden(benchmark, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert benchmark.resolve_dataset("synthetic", str(tmp_path), allow_download=True) == "synthetic"
    assert benchmark.resolve_dataset("cifar10", str(tmp_path), allow_download=False) == "cifar10"


# ---------------------------------------------------------------------------
# Failing usefully
# ---------------------------------------------------------------------------


def test_missing_data_fails_immediately_and_says_what_to_run(benchmark, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Silence for five minutes then a timeout is not an error message."""

    with pytest.raises(FileNotFoundError) as caught:
        benchmark.load_few_shot_split(
            dataset_name="cifar10", data_dir=str(tmp_path), allow_download=False
        )

    message = str(caught.value)
    assert "--allow-download" in message, "the message must name the flag that fixes it"
    assert "170MB" in message, "the reader should know what they are agreeing to"


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------


def test_the_fixture_is_deterministic(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Two runs at the same seed must produce identical tensors."""

    first, _ = benchmark._synthetic_split(n_way=3, k_shot=4, seed=42)
    second, _ = benchmark._synthetic_split(n_way=3, k_shot=4, seed=42)

    assert len(first) == len(second) == 12
    for (a_img, a_label), (b_img, b_label) in zip(first, second, strict=True):
        assert a_label == b_label
        assert a_img.equal(b_img)


def test_the_fixture_covers_every_class(benchmark) -> None:  # type: ignore[no-untyped-def]
    support, query = benchmark._synthetic_split(n_way=5, k_shot=10, seed=42)

    assert {label for _, label in support} == set(range(5))
    assert {label for _, label in query} == set(range(5))
    assert all(image.shape == (3, 32, 32) for image, _ in support)


# ---------------------------------------------------------------------------
# Not publishing a meaningless number
# ---------------------------------------------------------------------------


def test_no_accuracy_is_reported_for_synthetic_data(benchmark) -> None:  # type: ignore[no-untyped-def]
    """A figure measured on random tensors describes nothing.

    Publishing one would be the same mistake #17 had to retract -- worse,
    because it would look like a measurement rather than a claim.
    """

    from adaptshot.config.settings import AdaptShotConfig

    config = AdaptShotConfig(backbone="resnet18", device="cpu", seed=42, n_way=2, k_shot=2)
    results = benchmark.run_smoke_test(config, dataset="synthetic")

    assert results["accuracy"] is None
    assert results["data_source"] == "synthetic"
    assert results["latency_avg_ms"] > 0, "latency is measurable on any input, and is reported"


def test_the_determinism_check_uses_the_dataset_that_was_measured() -> None:
    """It used to always ask for CIFAR-10, whatever the benchmark actually ran on.

    Harmless while every path downloaded CIFAR anyway; a hard failure the moment
    downloads stopped being implicit. Caught by the offline CI job on the very
    PR that added it, which is the argument for having the job.

    Checked by reading the source rather than by running: exercising it needs a
    full embedding pass, and the property is that one name is threaded through.
    """

    source = (REPO_ROOT / "benchmarks" / "run_benchmark.py").read_text(encoding="utf-8")
    marker = "def run_once() -> np.ndarray:"
    assert marker in source
    body = source[source.index(marker) : source.index(marker) + 900]
    assert "dataset_name=dataset_name" in body, (
        "the determinism check must load the dataset that was benchmarked"
    )
    assert "allow_download=args.allow_download" in body, (
        "and must respect the same download permission"
    )


def test_the_committed_artifact_came_from_real_data() -> None:
    """`results/smoke_test.json` is quoted in the documentation.

    The offline fixture makes it easy to regenerate it accidentally, which would
    replace a measured figure with `null` or, worse, leave a stale number beside
    a synthetic provenance. Whatever is committed must say where it came from,
    and that must not be the fixture.
    """

    if not ARTIFACT.is_file():
        pytest.skip("no committed benchmark artifact")

    results = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert results.get("data_source") != "synthetic", (
        "a synthetic benchmark artifact has been committed; regenerate with "
        "--dataset cifar10 --allow-download"
    )
    assert results.get("accuracy") is not None, (
        "the committed artifact has no accuracy figure, but the documentation "
        "quotes one"
    )
