"""Tests for FewShotLearner checkpoint persistence and migration."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from PIL import Image

from src.adaptshot import AdaptShotConfig, AdaptShotError, FewShotLearner


@pytest.fixture
def support_images(tmp_path: Path) -> list[str]:
    paths: list[str] = []
    for index, color in enumerate(["red", "green"]):
        path = tmp_path / f"support_{index}.png"
        Image.new("RGB", (32, 32), color=color).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def patched_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract(image: Any, config: AdaptShotConfig, return_numpy: bool = True, cache: Any = None) -> np.ndarray:
        if isinstance(image, Image.Image):
            color = image.getpixel((0, 0))
        else:
            color = (0, 0, 0)
        base = float(sum(color)) / 255.0
        embedding = np.full(8, base, dtype=np.float32)
        if return_numpy:
            return embedding
        return embedding

    monkeypatch.setattr("src.adaptshot.core.learner.extract_embedding", fake_extract)


def _load_raw_state(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_raw_state(path: Path, state: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def test_save_load_roundtrip(
    tmp_path: Path,
    support_images: list[str],
    patched_embeddings: None,
) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
    learner.load_support_images(support_images, [0, 1])

    checkpoint = tmp_path / "checkpoint.json"
    learner.save(str(checkpoint))

    state = _load_raw_state(checkpoint)
    assert state["schema_version"] == "0.1.1"
    assert "integrity" in state
    assert state["integrity"]["checksum_sha256"]

    restored = FewShotLearner.load(str(checkpoint))

    assert restored._is_initialized is True
    assert restored._sim_labels == learner._sim_labels
    assert restored._sim_uncertainties == learner._sim_uncertainties
    assert restored._sim_access_times == learner._sim_access_times
    assert restored._model_head is not None


def test_corrupted_file_handling(tmp_path: Path, support_images: list[str], patched_embeddings: None) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
    learner.load_support_images(support_images, [0, 1])

    checkpoint = tmp_path / "corrupted.json"
    learner.save(str(checkpoint))
    emb_path = checkpoint.with_suffix(".embeddings.npy")
    emb_path.write_bytes(b"not-a-valid-npy-file")

    with pytest.raises(AdaptShotError, match="corrupted"):
        FewShotLearner.load(str(checkpoint))


def test_version_migration(tmp_path: Path, support_images: list[str], patched_embeddings: None) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
    learner.load_support_images(support_images, [0, 1])

    checkpoint = tmp_path / "legacy.json"
    learner.save(str(checkpoint))
    state = _load_raw_state(checkpoint)
    state.pop("schema_version", None)
    state.pop("integrity", None)
    state["buffer"].pop("previews", None)
    _write_raw_state(checkpoint, state)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = FewShotLearner.load(str(checkpoint))

    assert restored._is_initialized is True
    assert any("migrat" in str(item.message).lower() for item in caught)
    assert restored._sim_labels == learner._sim_labels


def test_integrity_hash_verification(tmp_path: Path, support_images: list[str], patched_embeddings: None) -> None:
    learner = FewShotLearner(config=AdaptShotConfig(device="cpu"))
    learner.load_support_images(support_images, [0, 1])

    checkpoint = tmp_path / "integrity.json"
    learner.save(str(checkpoint))

    state = _load_raw_state(checkpoint)
    state["config"]["k_shot"] = 99
    _write_raw_state(checkpoint, state)

    with pytest.raises(AdaptShotError, match="integrity"):
        FewShotLearner.load(str(checkpoint))
