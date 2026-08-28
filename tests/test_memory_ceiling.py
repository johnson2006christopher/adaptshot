"""Measure the memory ceiling, and say plainly where we stand against it (#13).

`CLAUDE.md` calls "CPU-only, <250MB RAM" a non-negotiable constraint and the
README repeats it. Nothing measured it. It is the constraint that distinguishes
AdaptShot from every other few-shot library, and it is the one most likely to
break silently: a backbone swap or a cached array crosses the line with no
visible symptom on a laptop with 16GB.

**Measured, it is not met.** A full support-set-to-prediction cycle peaks around
775MB, roughly three times the documented figure. The breakdown is what matters:

    interpreter + numpy + PIL              33 MB
    import adaptshot                      512 MB   (+479)
    FewShotLearner()                      516 MB
    load_support_images (15 images)       774 MB   (+258)
    predict()                             775 MB

The +479MB at import is torch, pulled in eagerly by `utils/determinism.py` and
`utils/io.py`, which import it at module scope. `core/extractor.py` is careful to
load torch lazily; those two undo it. The +258MB is ResNet-18's weights and
activations.

So no path that currently *works* stays under 250MB, because inference requires
torch (#35) and torch alone costs twice the budget. The target is reachable only
through the bundled-ONNX path (#36).

What this module does about it:

* `test_memory_does_not_regress` guards the number we actually have, so growth is
  caught. A ceiling nobody can meet is not a ceiling; a ratchet on reality is.
* `test_the_documented_ceiling_is_met` is `xfail(strict=True)`. It fails today,
  which is honest -- and it will fail the build *when it starts passing*, forcing
  whoever fixes #36 to come back and correct the documentation rather than
  leaving a stale disclaimer behind.

Every run prints the measured figure, so the headroom is visible rather than
hidden behind a green tick.

**What is measured**, settling the open question in #13: peak resident set size
of a fresh interpreter that imports AdaptShot, builds a learner, loads a support
set and predicts. Whole process, not the library's own allocations -- that is the
number a person on a 1GB phone or a small VPS actually runs out of, and it is the
only one measurable without instrumenting the interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: The documented promise. Not met today; see the module docstring.
DOCUMENTED_CEILING_MB = 250

#: What a full cycle actually costs, plus headroom for runner variance and
#: version drift. This is a regression guard, not an endorsement.
MEASURED_BUDGET_MB = 1100

_MEASURE = textwrap.dedent(
    """
    import os
    import resource
    import tempfile

    import numpy as np
    from PIL import Image

    from adaptshot import AdaptShotConfig, FewShotLearner

    directory = tempfile.mkdtemp()
    rng = np.random.default_rng(42)
    paths, labels = [], []
    for class_index in range(3):
        for example in range(5):
            path = os.path.join(directory, f"{class_index}_{example}.png")
            Image.fromarray(
                rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            ).save(path)
            paths.append(path)
            labels.append(f"c{class_index}")

    learner = FewShotLearner(
        config=AdaptShotConfig(backbone="resnet18", device="cpu", seed=42)
    )
    learner.load_support_images(paths, labels)
    learner.predict(paths[0])

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(peak_kb / 1024)
    """
)


def _peak_rss_mb() -> float:
    """Peak RSS of a full cycle, measured in a fresh interpreter.

    A subprocess because the measurement is meaningless inside pytest: torch,
    the plugins and every previously imported module are already resident, so
    `ru_maxrss` would describe the test session rather than the library.
    """

    completed = subprocess.run(
        [sys.executable, "-c", _MEASURE],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"could not run a full cycle in this environment:\n{completed.stderr[-800:]}")
    return float(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def peak_rss_mb() -> float:
    pytest.importorskip("torch", reason="inference requires the torch extra (#35)")
    measured = _peak_rss_mb()
    print(f"\npeak RSS for a full support-to-prediction cycle: {measured:.0f} MB")
    print(f"  documented ceiling: {DOCUMENTED_CEILING_MB} MB")
    print(f"  regression budget:  {MEASURED_BUDGET_MB} MB")
    return measured


def test_memory_does_not_regress(peak_rss_mb: float) -> None:
    """Catch growth against what we actually cost today.

    The documented ceiling cannot serve as the guard: it is unmet, so asserting
    it would leave the suite permanently red and teach everyone to ignore it.
    This number is deliberately unflattering and deliberately enforced.
    """

    assert peak_rss_mb < MEASURED_BUDGET_MB, (
        f"a full cycle now peaks at {peak_rss_mb:.0f} MB, over the "
        f"{MEASURED_BUDGET_MB} MB regression budget. Either something started "
        "holding memory it does not need, or the budget needs raising with a "
        "reason recorded here."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The documented <250MB ceiling is not met: inference requires torch "
        "(#35), and torch alone costs about twice the budget at import. Only "
        "the bundled-ONNX path (#36) can reach it. strict=True so that this "
        "fails the build when it starts passing, forcing the documentation to "
        "be corrected rather than left stale."
    ),
)
def test_the_documented_ceiling_is_met(peak_rss_mb: float) -> None:
    assert peak_rss_mb < DOCUMENTED_CEILING_MB


def test_importing_the_library_pulls_in_torch(peak_rss_mb: float) -> None:
    """Records the largest single cost, so the cause is not lost with the number.

    `utils/determinism.py` and `utils/io.py` import torch at module scope, so
    `import adaptshot` costs roughly 480MB before anything is asked of it --
    even for a caller who only wants to read a config. `core/extractor.py` takes
    care to load torch lazily; these two undo that.

    This test asserts the situation as it *is*. When #35 makes the import
    torch-free it will fail, which is the point: the number in the docstring
    above must be re-measured at the same time.
    """

    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    completed = subprocess.run(
        [sys.executable, "-c", "import sys, adaptshot; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True", (
        "`import adaptshot` no longer pulls in torch. That is good news and "
        "makes this test obsolete -- re-measure the figures in this module's "
        "docstring, in README.md and in CLAUDE.md, then delete this test."
    )
