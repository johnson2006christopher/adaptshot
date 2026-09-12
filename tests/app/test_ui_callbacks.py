"""The UI callbacks, exercised as functions (#108).

Fifty-three app tests passed while the dropdown refresh was broken, a
correction could land on another session's photograph, and the confidence
widget showed a string -- because nothing called the callbacks. These tests
call them.

Everything gradio-shaped skips without the `app` extra; the CLI's argument
contract is core behaviour and runs everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptshot.app import TambuaEngine, bundled_config
from tests.app.support.images import make_placeholder

# ---------------------------------------------------------------------------
# CLI argument contract (no gradio needed)
# ---------------------------------------------------------------------------


def test_share_without_auth_is_refused() -> None:
    """`--share` publishes the page to the internet; that needs a password.

    parser.error exits with code 2 before anything gradio-shaped is imported,
    so this holds on a core install too (#105).
    """

    from adaptshot.app.cli import launch

    with pytest.raises(SystemExit) as excinfo:
        launch(["--share"])
    assert excinfo.value.code == 2


def test_malformed_auth_is_refused() -> None:
    from adaptshot.app.cli import launch

    for bad in ("nopassword", "user:", ":pass"):
        with pytest.raises(SystemExit) as excinfo:
            launch(["--auth", bad])
        assert excinfo.value.code == 2, f"--auth {bad!r} should be rejected"


# ---------------------------------------------------------------------------
# Engine-level behaviour the UI depends on (no gradio needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> TambuaEngine:
    """An engine trained on deterministic images. Module-scoped: training is slow."""

    root: Path = tmp_path_factory.mktemp("support")
    engine = TambuaEngine(bundled_config("maize"))
    for key in engine.cfg.labels:
        folder = root / key
        folder.mkdir()
        for i in range(5):
            make_placeholder(key, variant=i).save(folder / f"{i:02d}.png")
    engine.load_images_from_dir(str(root))
    return engine


def test_full_set_counts_as_abstention(trained: TambuaEngine) -> None:
    """A set of every known class tells the user nothing; say so (#107).

    The `is_abstention` docstring always promised this; the code checked only
    the empty half until #107.
    """

    result = trained.identify(make_placeholder(trained.cfg.labels[0], variant=9))
    full = type(result)(
        **{
            **result.__dict__,
            "prediction_set": tuple(trained.known_labels),
            "known_classes": len(trained.known_labels),
        }
    )
    assert full.is_abstention
    empty = type(result)(**{**result.__dict__, "prediction_set": (), "known_classes": 3})
    assert empty.is_abstention
    partial = type(result)(
        **{
            **result.__dict__,
            "prediction_set": tuple(trained.known_labels[:2]),
            "known_classes": len(trained.known_labels),
        }
    )
    assert not partial.is_abstention


def test_identify_stamps_known_classes(trained: TambuaEngine) -> None:
    result = trained.identify(make_placeholder(trained.cfg.labels[0], variant=8))
    assert result.known_classes == len(trained.known_labels)


def test_teach_from_ui_corrects_the_named_image_not_the_last_one(
    trained: TambuaEngine, tmp_path: Path
) -> None:
    """Two sessions, one process: a correction names its own photograph (#104).

    Session A diagnoses photo A; session B diagnoses photo B (which makes B
    "the last image the process saw"); A submits a correction. It must land on
    A's photo.
    """

    labels = trained.cfg.labels
    photo_a = tmp_path / "a.png"
    photo_b = tmp_path / "b.png"
    make_placeholder(labels[0], variant=11).save(photo_a)
    make_placeholder(labels[1], variant=12).save(photo_b)

    trained.identify(str(photo_a))  # session A
    trained.identify(str(photo_b))  # session B -- overwrites _last_image_path

    message = trained.teach_from_ui(
        true_label=labels[0], confidence_weight=1.0, image_path=str(photo_a)
    )
    assert message.startswith("✅"), message
    assert trained.history.corrections[-1]["image_path"] == str(photo_a), (
        "the correction was applied to another session's photograph"
    )


def test_teach_grows_the_calibration_set(trained: TambuaEngine, tmp_path: Path) -> None:
    """The round-trip the Teach tab promises: correct once, calibration grows."""

    labels = trained.cfg.labels
    photo = tmp_path / "roundtrip.png"
    make_placeholder(labels[2], variant=13).save(photo)

    trained.identify(str(photo))
    before = int(trained.learner.conformal.calibration_size)
    trained.teach_from_ui(
        true_label=labels[2], confidence_weight=1.0, image_path=str(photo)
    )
    after = int(trained.learner.conformal.calibration_size)
    assert after > before, (
        f"a correction must add a calibration score: {before} -> {after}"
    )


# ---------------------------------------------------------------------------
# Gradio-level callbacks (need the app extra)
# ---------------------------------------------------------------------------


def test_the_app_builds() -> None:
    """`build_app()` must construct -- the smoke test the UI never had (#108)."""

    gr = pytest.importorskip("gradio", reason="the UI needs the app extra")
    from adaptshot.app import ui

    app = ui.build_app()
    assert isinstance(app, gr.Blocks)


def test_refresh_returns_a_dropdown_with_choices(trained: TambuaEngine) -> None:
    """The refresh button must update the *choices*, not set a list as value.

    Returning a bare list[str] made Dropdown.postprocess treat it as the
    selected value: the list never refreshed and the value became a Python
    list rendered as text (#107).
    """

    gr = pytest.importorskip("gradio", reason="the UI needs the app extra")
    from adaptshot.app import ui

    ui._state.engine = trained
    try:
        refreshed = ui._refreshed_dropdown()
        assert isinstance(refreshed, gr.Dropdown), (
            f"the refresh callback must return a Dropdown update, got "
            f"{type(refreshed).__name__}"
        )
        names = [
            choice[0] if isinstance(choice, (tuple, list)) else choice
            for choice in (refreshed.choices or [])
        ]
        for label in trained.known_labels:
            assert label in names, f"known label {label!r} missing from the dropdown"
    finally:
        ui._state.engine = None


def test_teach_callback_requires_an_image(trained: TambuaEngine) -> None:
    pytest.importorskip("gradio", reason="the UI needs the app extra")
    from adaptshot.app import ui

    ui._state.engine = trained
    try:
        message = ui._teach(trained.cfg.labels[0], 1.0, None)
        assert message.startswith("❌"), (
            "with no photo in this session the callback must refuse, "
            f"got: {message}"
        )
    finally:
        ui._state.engine = None
