"""The parts of the distribution-shift benchmark that hold without the dataset (#29).

The benchmark itself needs the 400-photo pool. What can be checked offline is
that the shifts do what they say -- each is a transform of a real image, each
identity level is the identity, each non-identity level changes pixels and
preserves size -- that the in-situ selection is class-balanced (the mistake
that cost coverage on the first run), and that the early-warning statistic is
the correlation it claims to be.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from benchmarks.run_shift import SUITE, _balanced, early_warning


@pytest.fixture
def photo() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(40, 220, (96, 128, 3), dtype=np.uint8))


@pytest.mark.parametrize("kind", list(SUITE))
def test_identity_level_leaves_the_image_alone(kind: str, photo: Image.Image) -> None:
    shift, levels = SUITE[kind]
    out = shift(photo, levels[0])
    assert out.size == photo.size and out.mode == "RGB"
    if kind != "jpeg":  # JPEG at quality 95 is still lossy; the others are exact
        assert np.array_equal(np.asarray(out), np.asarray(photo))


@pytest.mark.parametrize("kind", list(SUITE))
def test_every_other_level_changes_pixels_and_keeps_the_frame(kind: str, photo: Image.Image) -> None:
    shift, levels = SUITE[kind]
    for level in levels[1:]:
        out = shift(photo, level)
        assert out.size == photo.size and out.mode == "RGB"
        assert not np.array_equal(np.asarray(out), np.asarray(photo)), f"{kind} at {level} changed nothing"


def test_stronger_shift_moves_further_from_the_original(photo: Image.Image) -> None:
    """Levels are ordered by severity; the mean pixel change must not decrease."""

    base = np.asarray(photo, dtype=np.float64)
    for kind in ("blur", "downscale"):
        shift, levels = SUITE[kind]
        drift = [np.abs(np.asarray(shift(photo, lv), dtype=np.float64) - base).mean() for lv in levels]
        assert drift == sorted(drift), f"{kind}: {drift}"


def test_in_situ_selection_is_class_balanced() -> None:
    """Class-by-class layout, k=6 over three classes: two of each, not six of one."""

    labels = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5, dtype=object)
    indices = np.arange(15)
    chosen = _balanced(indices, labels, 6)
    counts = {name: int((labels[chosen] == name).sum()) for name in "abc"}
    assert counts == {"a": 2, "b": 2, "c": 2}
    assert len(set(chosen.tolist())) == 6


def test_early_warning_is_high_when_the_flag_tracks_the_loss() -> None:
    def cell(identity: bool, coverage: float, ood: float) -> dict:  # type: ignore[type-arg]
        return {"identity": identity, "coverage": {"mean": coverage}, "ood_rate": {"mean": ood}}

    tracking = [cell(True, 0.97, 0.02), cell(False, 0.95, 0.03), cell(False, 0.85, 0.10), cell(False, 0.75, 0.20)]
    silent = [cell(True, 0.97, 0.02), cell(False, 0.95, 0.02), cell(False, 0.85, 0.02), cell(False, 0.75, 0.02)]
    assert early_warning(tracking, 0.9)["correlation"] > 0.9
    assert early_warning(silent, 0.9)["correlation"] is None  # a flat flag rate has no correlation to report
