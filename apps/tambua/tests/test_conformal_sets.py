"""The app must present conformal sets, and must not overstate them (#48).

`ConformalEngine` was in AdaptShot's public `__all__` and used by the
application exactly zero times: the flagship demonstration did not demonstrate
the capability that most distinguishes AdaptShot from a plain classifier.

The trap in fixing that is subtler than the gap. Until enough calibration scores
accumulate, conformal returns the top label as a singleton and reports coverage
equal to `1 - alpha` -- the *target*, restated. Presenting that as a measurement
is exactly the mistake #17 cost a release to correct, so several tests here exist
only to make sure the interface never does it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tambua", reason="the application is not installed in this environment")

from support.images import make_placeholder
from tambua import ClassInfo, TambuaEngine, bundled_config, combined_action


def _info(key: str, action: str, local: str | None = None) -> ClassInfo:
    return ClassInfo(
        key=key,
        local_name=local or key,
        action=action,
        description="",
        severity="moderate",
        domain="d",
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> TambuaEngine:
    """An engine trained on deterministic images. Module-scoped: training is slow."""

    pytest.importorskip("torch", reason="inference needs the torch extra (#35)")
    root: Path = tmp_path_factory.mktemp("support")
    engine = TambuaEngine(bundled_config("maize"))
    for key in engine.cfg.labels:
        folder = root / key
        folder.mkdir()
        for i in range(5):
            make_placeholder(key, variant=i).save(folder / f"{i:02d}.png")
    engine.load_images_from_dir(str(root))
    return engine


# ---------------------------------------------------------------------------
# The set itself
# ---------------------------------------------------------------------------


def test_alpha_comes_from_the_config(trained: TambuaEngine) -> None:
    assert trained.cfg.engine.conformal_alpha == 0.1
    assert trained.learner.conformal.alpha == 0.1


def test_identification_carries_a_prediction_set(trained: TambuaEngine, tmp_path: Path) -> None:
    label = trained.cfg.labels[0]
    path = tmp_path / "q.png"
    make_placeholder(label, variant=99).save(path)

    result = trained.identify(str(path))
    assert result.prediction_set, "every non-abstaining result must carry a set"
    assert result.label in result.prediction_set, (
        "the top-1 label must always be inside its own set"
    )
    assert result.alpha == 0.1


def test_the_csv_export_carries_the_set(trained: TambuaEngine, tmp_path: Path) -> None:
    """A downstream reader must be able to see the set, not just the top label."""

    path = tmp_path / "q.png"
    make_placeholder(trained.cfg.labels[0], variant=98).save(path)

    csv = trained.batch_to_csv(trained.batch_identify([str(path)]))
    header = csv.splitlines()[0]
    for column in ("prediction_set", "set_size", "alpha", "coverage_measured"):
        assert column in header


# ---------------------------------------------------------------------------
# Not overstating the guarantee
# ---------------------------------------------------------------------------


def test_coverage_is_not_reported_as_measured_before_calibration() -> None:
    """`1 - alpha` is the target. Quoting it as a result is the #17 mistake."""

    from tambua.engine import Identification

    fresh = Identification(
        label="x", local_name="x", confidence=0.9, raw_confidence=0.9,
        action="", severity="low", ood_flag=False, uncertainty_flag=False,
        act_action="", distance_to_prototype=0.0, calibrated_ece=0.0,
        prediction_set=("x",), alpha=0.1,
        coverage_is_measured=False, empirical_coverage=0.0, calibration_size=2,
    )
    assert fresh.empirical_coverage == 0.0, (
        "an uncalibrated result must not carry a coverage figure at all"
    )


def test_the_interface_says_when_coverage_is_not_yet_measured() -> None:
    from tambua.app import _render_prediction_set
    from tambua.engine import Identification

    engine = TambuaEngine(bundled_config("maize"))
    label = engine.cfg.labels[0]
    uncalibrated = Identification(
        label=label, local_name=label, confidence=0.9, raw_confidence=0.9,
        action="", severity="low", ood_flag=False, uncertainty_flag=False,
        act_action="", distance_to_prototype=0.0, calibrated_ece=0.0,
        prediction_set=(label,), alpha=0.1,
        coverage_is_measured=False, empirical_coverage=0.0, calibration_size=3,
    )
    rendered = _render_prediction_set(engine, uncalibrated, "🟢")

    assert "Not yet calibrated" in rendered
    assert "90%" not in rendered, (
        "the target coverage must not appear while it is unmeasured -- a reader "
        "cannot tell an aspiration from a measurement once it is a percentage"
    )


def test_a_multi_member_set_is_presented_as_a_set_not_a_winner() -> None:
    """A person reading a top-1 has stopped reading before the caveat arrives."""

    from tambua.app import _render_prediction_set
    from tambua.engine import Identification

    engine = TambuaEngine(bundled_config("maize"))
    members = tuple(sorted(engine.cfg.labels[:2]))
    result = Identification(
        label=members[0], local_name=members[0], confidence=0.6, raw_confidence=0.6,
        action="", severity="high", ood_flag=False, uncertainty_flag=False,
        act_action="", distance_to_prototype=0.0, calibrated_ece=0.0,
        prediction_set=members, alpha=0.1,
        coverage_is_measured=True, empirical_coverage=0.91, calibration_size=40,
    )
    rendered = _render_prediction_set(engine, result, "🟠")

    assert "One of these 2" in rendered
    for member in members:
        assert engine.label_to_info(member).local_name in rendered
    assert "91%" in rendered


# ---------------------------------------------------------------------------
# Advice across a set
# ---------------------------------------------------------------------------


def test_matching_actions_across_a_set_give_one_instruction() -> None:
    """The most useful case conformal produces: ambiguous, but it does not matter."""

    advice = combined_action([
        _info("a", "Apply fungicide early."),
        _info("b", "Apply fungicide early."),
    ])
    assert "All 2 possibilities call for the same thing" in advice
    assert "Apply fungicide early." in advice


def test_conflicting_actions_are_stated_not_resolved() -> None:
    """Picking one would invent a recommendation nobody wrote, invisibly."""

    advice = combined_action([
        _info("a", "Remove the affected plants.", local="ya kwanza"),
        _info("b", "Do nothing; continue as normal.", local="ya pili"),
    ])
    assert "different things" in advice
    assert "Remove the affected plants." in advice
    assert "Do nothing; continue as normal." in advice


def test_an_empty_set_routes_to_a_human() -> None:
    advice = combined_action([])
    assert "someone who can look at it" in advice


def test_abstention_is_rendered_as_abstention() -> None:
    from tambua.app import _render_prediction_set
    from tambua.engine import Identification

    engine = TambuaEngine(bundled_config("maize"))
    abstained = Identification(
        label="?", local_name="?", confidence=0.2, raw_confidence=0.2,
        action="", severity="low", ood_flag=True, uncertainty_flag=True,
        act_action="", distance_to_prototype=0.0, calibrated_ece=0.0,
        prediction_set=(), alpha=0.1,
        coverage_is_measured=True, empirical_coverage=0.9, calibration_size=40,
    )
    assert abstained.is_abstention
    assert "Not confident enough" in _render_prediction_set(engine, abstained, "⚪")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.1, "half", True])
def test_alpha_outside_the_open_unit_interval_is_rejected(
    tmp_path: Path, bad: object
) -> None:
    """alpha=0 asks for every class; alpha=1 asks for no guarantee. Neither is a setting."""

    from tambua.config import load_config

    from adaptshot.utils.exceptions import ConfigValidationError

    source = Path(bundled_config("maize")).read_text(encoding="utf-8")
    value = f'"{bad}"' if isinstance(bad, str) else str(bad).lower()
    target = tmp_path / "bad.yaml"
    target.write_text(source.replace("conformal_alpha: 0.1", f"conformal_alpha: {value}"))

    with pytest.raises(ConfigValidationError) as caught:
        load_config(str(target))
    assert "conformal_alpha" in str(caught.value)
