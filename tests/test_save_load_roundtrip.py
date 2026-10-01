"""A loaded learner must answer exactly like the one that was saved (#118).

The measured round trip on 0.3.0: conformal calibration 24 scores -> 0, OOD
threshold 26.15 -> inf, contrastive projection unfitted, calibration window
emptied. A tomato leaf that was flagged OOD with a three-class set before the
save came back `ood_flag=False` with a confident singleton after the load.

`load()` now RECALIBRATES: everything derivable from the restored support
buffer — the LOO conformal scores, the OOD Gaussians, the contrastive
projection, the calibration window — is refit deterministically, and the
persisted temperature (fitted on real outcomes) is applied on top. These
tests compare full `PredictionResult`s across the round trip, on an
in-distribution photograph and on the foreign one that exposed the bug.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.data import demo_images, sample_images


@pytest.fixture()
def trained() -> FewShotLearner:
    """Calibrated at alpha=0.10 so the conformal buffer is live (floor 9)."""

    paths, labels = sample_images()
    learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
    learner.load_support_images(paths, labels)
    return learner


def _roundtrip(learner: FewShotLearner, tmp_path: Path) -> FewShotLearner:
    learner.save(str(tmp_path / "ckpt.json"))
    return FewShotLearner.load(str(tmp_path / "ckpt.json"))


def test_prediction_results_survive_the_round_trip(
    trained: FewShotLearner, tmp_path: Path
) -> None:
    paths, _ = sample_images()
    probes = [paths[0], demo_images()[-1]]  # one maize leaf, one foreign image

    before = [trained.predict(p) for p in probes]
    loaded = _roundtrip(trained, tmp_path)
    after = [loaded.predict(p) for p in probes]

    for probe, first, second in zip(probes, before, after, strict=True):
        a, b = dataclasses.asdict(first), dataclasses.asdict(second)
        for field in (
            "prediction",
            "conformal_set",
            "conformal_calibrated",
            "ood_flag",
            "uncertainty_flag",
            "act_action",
        ):
            assert a[field] == b[field], (
                f"{field} changed across save/load for {Path(probe).name}: "
                f"{a[field]!r} -> {b[field]!r}"
            )
        assert a["calibrated_confidence"] == pytest.approx(
            b["calibrated_confidence"], abs=1e-6
        )


def test_the_calibration_state_is_rebuilt_not_dropped(
    trained: FewShotLearner, tmp_path: Path
) -> None:
    scores_before = len(trained.conformal._calibration_scores)
    window_before = len(trained.calibrator._window_confidences)
    threshold_before = trained._ood_distance_threshold
    assert scores_before > 0 and window_before > 0

    loaded = _roundtrip(trained, tmp_path)

    assert len(loaded.conformal._calibration_scores) == scores_before, (
        "the conformal calibration buffer was dropped on load"
    )
    assert len(loaded.calibrator._window_confidences) == window_before, (
        "the calibration window was dropped on load"
    )
    assert np.isfinite(loaded._ood_distance_threshold), (
        "the OOD threshold came back non-finite"
    )
    assert loaded._ood_distance_threshold == pytest.approx(threshold_before, rel=1e-6)


def test_the_persisted_temperature_wins_over_the_bootstrap_refit(
    trained: FewShotLearner, tmp_path: Path
) -> None:
    """The saved temperature was fitted on real outcomes; the bootstrap's
    refit on LOO data must not overwrite it during load."""

    trained.calibrator.temperature = 1.2345  # as if fitted from corrections
    loaded = _roundtrip(trained, tmp_path)
    assert loaded.calibrator.temperature == pytest.approx(1.2345)


def test_contrastive_mode_is_fitted_after_load(tmp_path: Path) -> None:
    paths, labels = sample_images()
    learner = FewShotLearner(
        config=AdaptShotConfig(conformal_alpha=0.10, inference_mode="contrastive")
    )
    learner.load_support_images(paths, labels)
    assert learner.contrastive.is_fitted

    loaded = _roundtrip(learner, tmp_path)
    assert loaded.contrastive.is_fitted, (
        "a loaded contrastive learner silently fell back to nearest-neighbour"
    )
    probe = paths[0]
    assert dataclasses.asdict(loaded.predict(probe))["prediction"] == (
        dataclasses.asdict(learner.predict(probe))["prediction"]
    )


def test_act_thresholds_beyond_the_preallocated_slots_survive(
    trained: FewShotLearner, tmp_path: Path
) -> None:
    """A fresh engine preallocates max(10, n_way) slots; index 17 used to be
    dropped silently on load."""

    trained.act.record_outcome(17, correct=False)
    saved_threshold = trained.act.get_threshold(17)
    assert saved_threshold != pytest.approx(0.65)

    loaded = _roundtrip(trained, tmp_path)
    assert loaded.act.get_threshold(17) == pytest.approx(saved_threshold), (
        "the eleventh class's ACT threshold was dropped on load"
    )


def test_a_v011_checkpoint_loads_with_a_migration_warning(
    trained: FewShotLearner, tmp_path: Path
) -> None:
    """0.1.1 is the schema the 0.1.0 migrator *produces*; rejecting it meant
    the migrator's own output could not be loaded."""

    trained.save(str(tmp_path / "ckpt.json"))
    payload = json.loads((tmp_path / "ckpt.json").read_text(encoding="utf-8"))
    payload["schema_version"] = "0.1.1"
    payload.pop("integrity", None)  # 0.1.x checkpoints shipped no hash
    (tmp_path / "ckpt.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="migrating"):
        loaded = FewShotLearner.load(str(tmp_path / "ckpt.json"))
    assert loaded.support_size == trained.support_size
