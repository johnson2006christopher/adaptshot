"""Guards the torch-optional contract advertised in the README.

AdaptShot promises that the core install (`numpy` + `Pillow`) is enough to
import the library and run inference — PyTorch is only needed for training.
Nothing enforced that promise until this module existed, and it silently broke:
an annotation evaluated at import time referenced ``nn.Module`` while ``nn``
was ``None``, making ``import adaptshot`` fail on every Python before 3.14.

These tests run in a subprocess with ``torch`` blocked at the import-system
level, so they hold whether or not torch is installed in the test environment.

Note: on Python >= 3.14 annotations are lazily evaluated (PEP 649), which masks
this specific class of bug. The CI matrix covers 3.9-3.12, where it is caught.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Installs a meta-path hook that makes any `import torch` raise ImportError,
# simulating a core-only install even when torch is present.
_BLOCK_TORCH_PREAMBLE = """
import sys

class _TorchBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch is blocked: simulating a core-only install")
        return None

sys.meta_path.insert(0, _TorchBlocker())
"""


def _run_without_torch(body: str) -> subprocess.CompletedProcess[str]:
    """Execute ``body`` in a fresh interpreter where torch cannot be imported."""

    script = _BLOCK_TORCH_PREAMBLE + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,  # the caller asserts on returncode
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_package_imports_without_torch() -> None:
    """`import adaptshot` must succeed with only numpy and Pillow installed."""

    result = _run_without_torch(
        """
        from adaptshot import FewShotLearner, AdaptShotConfig  # noqa: F401
        print("IMPORT_OK")
        """
    )
    assert result.returncode == 0, (
        "importing adaptshot without torch failed — this breaks the core install "
        f"promised in the README.\n\nstderr:\n{result.stderr}"
    )
    assert "IMPORT_OK" in result.stdout


def test_learner_constructs_without_torch() -> None:
    """A CPU learner must be constructible on a core-only install."""

    result = _run_without_torch(
        """
        from adaptshot import AdaptShotConfig, FewShotLearner

        learner = FewShotLearner(config=AdaptShotConfig(backbone="resnet18", device="cpu"))
        assert learner is not None
        print("LEARNER_OK")
        """
    )
    assert result.returncode == 0, (
        f"constructing FewShotLearner without torch failed.\n\nstderr:\n{result.stderr}"
    )
    assert "LEARNER_OK" in result.stdout


def test_public_api_exports_are_importable_without_torch() -> None:
    """Every name in ``__all__`` must resolve on a core-only install."""

    result = _run_without_torch(
        """
        import adaptshot as adaptshot

        missing = [name for name in adaptshot.__all__ if not hasattr(adaptshot, name)]
        assert not missing, f"unresolvable exports without torch: {missing}"
        print("EXPORTS_OK", len(adaptshot.__all__))
        """
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "EXPORTS_OK" in result.stdout


# ---------------------------------------------------------------------------
# The boundary itself (#35)
# ---------------------------------------------------------------------------
#
# The tests above prove the package *imports* without torch. They do not prove
# it *works* without torch, and that gap is exactly where #35 lived: `predict`,
# `correct` and save/load all raised `ModuleNotFoundError` while these passed.
#
# So this section calls things. It records the boundary as it actually is, and
# asserts it in both directions -- a capability that stops working fails, and so
# does one that starts working, because either way the README has to change.

#: What a core install can do today, measured by calling each one with torch
#: blocked. The interesting part is what is on this list: the calibration,
#: conformal and uncertainty maths -- the parts that distinguish AdaptShot from
#: a plain classifier -- are already pure numpy.
TORCH_FREE_OPERATIONS = (
    "AdaptShotConfig()",
    "FewShotLearner()",
    "set_deterministic_seed()",
    "ConformalEngine()",
    "UncertaintyQuantifier.quantify()",
    "calibration_report()",
)

#: What a core install cannot do. One entry, and it is the one that matters:
#: turning an image into an embedding. Everything downstream of an embedding
#: already works without torch, which is what makes the bundled-ONNX path (#36)
#: the whole of the remaining distance rather than a first step.
TORCH_REQUIRED_OPERATIONS = ("load_support_images()",)

_PROBE = '''
import os
import tempfile

import numpy as np
from PIL import Image

results = {}


def probe(name, fn):
    try:
        fn()
    except ModuleNotFoundError:
        results[name] = "needs-torch"
    except Exception as exc:  # noqa: BLE001 - classifying, not handling
        results[name] = type(exc).__name__
    else:
        results[name] = "works"


from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.core.conformal import ConformalEngine
from adaptshot.core.uncertainty import UncertaintyQuantifier
from adaptshot.utils.determinism import set_deterministic_seed

probe("AdaptShotConfig()", lambda: AdaptShotConfig(backbone="resnet18", device="cpu"))
config = AdaptShotConfig(backbone="resnet18", device="cpu", seed=42)
probe("FewShotLearner()", lambda: FewShotLearner(config=config))
probe("set_deterministic_seed()", lambda: set_deterministic_seed(42))
probe("ConformalEngine()", ConformalEngine)
probe(
    "UncertaintyQuantifier.quantify()",
    lambda: UncertaintyQuantifier().quantify(
        np.ones(64, dtype=np.float32),
        np.ones((4, 64), dtype=np.float32),
        np.array(["a"] * 4),
    ),
)

learner = FewShotLearner(config=config)
probe("calibration_report()", learner.calibration_report)

directory = tempfile.mkdtemp()
rng = np.random.default_rng(42)
paths, labels = [], []
for class_index in range(2):
    for example in range(3):
        path = os.path.join(directory, f"{class_index}_{example}.png")
        Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(path)
        paths.append(path)
        labels.append(f"c{class_index}")

probe("load_support_images()", lambda: learner.load_support_images(paths, labels))

for name, outcome in results.items():
    print(f"{name}\\t{outcome}")
'''


def _probe_boundary() -> dict[str, str]:
    completed = _run_without_torch(_PROBE)
    assert completed.returncode == 0, (
        "the boundary probe itself failed to run:\n" + completed.stderr[-2000:]
    )
    return dict(
        line.split("\t", 1) for line in completed.stdout.strip().splitlines() if "\t" in line
    )


def test_the_torch_free_operations_still_work() -> None:
    """Everything downstream of an embedding runs on a core install.

    If one of these regresses, the core install has lost a capability the
    README describes, and that is a bug rather than a documentation change.
    """

    outcomes = _probe_boundary()
    broken = {
        name: outcomes.get(name, "missing")
        for name in TORCH_FREE_OPERATIONS
        if outcomes.get(name) != "works"
    }
    assert not broken, f"these worked without torch and no longer do: {broken}"


def test_the_torch_required_operations_still_require_torch() -> None:
    """Asserted in this direction on purpose.

    When #36 lands and embeddings come from a bundled ONNX backbone, this test
    fails -- which is correct. `README.md:100` says "PyTorch is optional and
    needed only for training", and that sentence becomes true at the same
    moment. Whoever makes it true has to come here and say so.
    """

    outcomes = _probe_boundary()
    unexpectedly_working = [
        name for name in TORCH_REQUIRED_OPERATIONS if outcomes.get(name) == "works"
    ]
    assert not unexpectedly_working, (
        f"{unexpectedly_working} now works without torch. Good news -- update "
        "TORCH_REQUIRED_OPERATIONS, README.md:100 and README.md:109-110, which "
        "currently describe a torch-free core that did not exist (#35)."
    )


def test_seeding_works_without_torch() -> None:
    """`set_deterministic_seed` is the function CLAUDE.md tells everyone to call.

    `utils/determinism.py` imported torch at module scope, so on a core install
    the one reproducibility helper contributors are instructed to use could not
    be imported at all. Seeding Python and NumPy is useful with or without
    torch; doing it only when torch happened to be installed was never intended.
    """

    completed = _run_without_torch(
        """
        from adaptshot.utils.determinism import set_deterministic_seed, verify_determinism
        import numpy as np

        set_deterministic_seed(42)
        assert verify_determinism(lambda: np.array([1.0, 2.0]))
        print("ok")
        """
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "ok" in completed.stdout
