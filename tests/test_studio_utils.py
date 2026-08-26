"""Tests for AdaptShot Studio helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

# Unlike `core/extractor.py`, `studio/utils.py` imports torch eagerly at module
# scope (line 25), so `adaptshot.studio` itself is unimportable without it. Studio
# lives behind the `gui` extra and is not part of the torch-free core contract, so
# the guard must precede the imports below rather than sit after them.
pytest.importorskip("torch")

from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.learner import FewShotLearner
from adaptshot.studio import utils as studio_utils
from adaptshot.studio.utils import (
    build_report_markdown,
    collect_image_sources,
    discover_images_in_folder,
    export_deployment_bundle,
    load_project_bundle,
)


def test_discover_images_in_folder_recursively(tmp_path: Path) -> None:
    """Ensure folder discovery finds nested image files."""

    root = tmp_path / "dataset"
    class_a = root / "class_a"
    nested = class_a / "nested"
    nested.mkdir(parents=True)
    image_a = class_a / "a.png"
    image_b = nested / "b.jpg"
    Image.new("RGB", (16, 16), color="red").save(image_a)
    Image.new("RGB", (16, 16), color="blue").save(image_b)

    discovered = discover_images_in_folder(root)

    assert discovered == sorted([image_a, image_b])


def test_collect_image_sources_deduplicates_file_and_folder(tmp_path: Path) -> None:
    """Ensure mixed uploads and folder imports stay unique and sorted."""

    root = tmp_path / "dataset"
    root.mkdir()
    image_a = root / "a.png"
    image_b = root / "b.png"
    Image.new("RGB", (16, 16), color="red").save(image_a)
    Image.new("RGB", (16, 16), color="blue").save(image_b)

    collected = collect_image_sources([image_a], folder_text=str(root))

    assert collected == sorted([image_a.resolve(), image_b.resolve()])


def test_export_bundle_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure exported studio bundles can be reopened later."""

    exports_root = tmp_path / "exports"
    monkeypatch.setattr(studio_utils, "EXPORT_ROOT", exports_root)

    support_dir = tmp_path / "support"
    support_dir.mkdir()
    image_a = support_dir / "a.png"
    image_b = support_dir / "b.png"
    Image.new("RGB", (32, 32), color="red").save(image_a)
    Image.new("RGB", (32, 32), color="blue").save(image_b)

    learner = FewShotLearner(config=AdaptShotConfig(backbone="mobilenet_v3_small", device="cpu", seed=7))
    learner.load_support_images([str(image_a), str(image_b)], ["class_a", "class_b"])

    archive_path = export_deployment_bundle(learner, "torchscript")
    reopened = load_project_bundle(archive_path)

    assert archive_path.exists()
    assert archive_path.suffix == ".zip"
    assert len(getattr(reopened, "_sim_embeddings", [])) == 2


def test_report_uses_benchmark_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure export reports show real benchmark data when it is available."""

    exports_root = tmp_path / "exports"
    monkeypatch.setattr(studio_utils, "EXPORT_ROOT", exports_root)

    support_dir = tmp_path / "support"
    support_dir.mkdir()
    image_a = support_dir / "a.png"
    image_b = support_dir / "b.png"
    Image.new("RGB", (32, 32), color="red").save(image_a)
    Image.new("RGB", (32, 32), color="blue").save(image_b)

    learner = FewShotLearner(config=AdaptShotConfig(backbone="mobilenet_v3_small", device="cpu", seed=7))
    learner.load_support_images([str(image_a), str(image_b)], ["class_a", "class_b"])

    session = studio_utils.StudioSession()
    session.benchmark_snapshot = {
        "accuracy": 1.0,
        "latency_avg_s": 0.012,
        "latency_p95_s": 0.019,
        "joules_estimate": 0.8,
        "co2_g_estimate": 0.0002,
    }

    report = build_report_markdown(learner, session=session)

    assert "Benchmark accuracy: 1.0000" in report
    assert "Benchmark energy estimate: 0.8000 J" in report
