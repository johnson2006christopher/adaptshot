"""The library never downloads silently, and never unpickles blindly (#121).

Three promises pinned here: a torch backbone whose weights are not cached
raises a `BackboneError` naming `allow_download` instead of fetching ~45 MB
mid-prediction; the fine-tune head checkpoint is loaded with
`weights_only=True`, so a crafted `.head.pt` cannot execute code on load; and
a structurally broken checkpoint surfaces as `AdaptShotError`, not a bare
`KeyError` from three frames deep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.data import sample_images
from adaptshot.utils.exceptions import AdaptShotError, BackboneError


def test_uncached_torch_weights_raise_instead_of_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from adaptshot.core import extractor

    # Simulate a cold cache without touching the real one.
    monkeypatch.setattr(extractor, "_torch_weights_cached", lambda name: False)
    extractor.clear_backbone_cache()
    try:
        with pytest.raises(BackboneError, match="allow_download"):
            extractor._build_backbone("resnet18", "cpu", allow_download=False)
    finally:
        extractor.clear_backbone_cache()


def test_allow_download_defaults_to_false() -> None:
    assert AdaptShotConfig().allow_download is False, (
        "silent downloads must be opt-in; a library for offline use cannot "
        "fetch 45 MB mid-prediction by default"
    )


def test_head_checkpoint_is_loaded_weights_only() -> None:
    """The one unguarded deserialisation input (#121), read from the source.

    Asserted statically rather than by crafting a malicious pickle: the test
    must hold on every install, and torch >= 2.6 flips the default anyway —
    the explicit argument is what guarantees it on the 2.0 floor.
    """

    source = (
        Path(__file__).resolve().parents[1] / "src" / "adaptshot" / "core" / "learner.py"
    ).read_text(encoding="utf-8")
    load_calls = list(source.split("_get_torch().load(")[1:])
    assert load_calls, "the head-loading call disappeared; move this test with it"
    for chunk in load_calls:
        assert "weights_only=True" in chunk[:300], (
            "a torch.load call without weights_only=True: before torch 2.6 "
            "that unpickles arbitrary objects from disk"
        )


def test_malformed_checkpoint_raises_a_named_error(tmp_path: Path) -> None:
    paths, labels = sample_images()
    learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
    learner.load_support_images(paths[:3], labels[:3])
    learner.save(str(tmp_path / "ckpt.json"))

    payload = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
    # Structural damage the integrity hash cannot see: it covers config and
    # embeddings, and the damage is elsewhere. Two layers answer: a missing
    # buffer field hits the existing explicit validation; a mistyped scalar
    # deeper in falls through to the #121 wrapper. Both must be AdaptShotError.
    del payload["buffer"]["labels"]
    (tmp_path / "ckpt.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AdaptShotError, match="buffer lengths"):
        FewShotLearner.load(str(tmp_path / "ckpt.json"))


def test_a_mistyped_checkpoint_scalar_raises_a_named_error(tmp_path: Path) -> None:
    paths, labels = sample_images()
    learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
    learner.load_support_images(paths[:3], labels[:3])
    learner.save(str(tmp_path / "ckpt.json"))

    payload = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
    payload["calibration"]["temperature"] = "hot"  # float("hot") three frames deep
    (tmp_path / "ckpt.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AdaptShotError, match="malformed"):
        FewShotLearner.load(str(tmp_path / "ckpt.json"))


def test_unknown_config_key_raises_a_named_error(tmp_path: Path) -> None:
    paths, labels = sample_images()
    learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
    learner.load_support_images(paths[:3], labels[:3])
    learner.save(str(tmp_path / "ckpt.json"))

    payload = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
    payload["config"]["a_key_from_the_future"] = 7
    payload.pop("integrity", None)
    (tmp_path / "ckpt.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AdaptShotError, match="malformed"):
        FewShotLearner.load(str(tmp_path / "ckpt.json"))
