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
