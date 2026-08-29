"""Does the coverage guarantee survive the field? Conformal under distribution shift (#29).

Split conformal promises P(true class in set) >= 1 - alpha **under
exchangeability**: calibration and test drawn from the same distribution. A
field photograph is not drawn from the distribution of the support set it is
compared against -- different phone, light, season, focus -- and the library
has no way to know. So this measures what happens to coverage when the queries
shift and the support does not, and whether anything the library already
computes would have warned.

Shifts are applied to real PlantVillage photographs, queries only, with
Pillow: Gaussian blur, brightness, JPEG re-compression and downscale-upscale.
Each is a transform of a real photograph, not a drawn one, and each has a
level 0 that is the identity, so the clean figure is measured in the same run.

Three questions, in the order the issue asks them:

1. **Coverage versus shift magnitude.** The curve itself is the result.
2. **Does the OOD detector warn first?** The flag rate per cell, and its
   correlation with the coverage lost.
3. **Does the human-in-the-loop path recover it?** After measuring, a handful
   of the *shifted* calibration photographs are fed through
   ``FewShotLearner.correct()`` -- the library's existing correction path,
   which updates the replay buffer and the conformal calibration -- and the
   shifted queries are measured again.

Runs on the core install, deliberately: ``correct()`` also triggers CA-EWC
fine-tuning when torch is importable, and the mitigation under test is
recalibration from in-situ labels, not fine-tuning.

    python -m benchmarks.run_shift --seed 42

Writes results/plantvillage_shift.json.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.utils.determinism import set_deterministic_seed
from benchmarks.plantvillage import (
    DatasetMissing,
    Episode,
    dataset_provenance,
    load_pool,
    sample_episodes,
)
from benchmarks.run_plantvillage import hardware, mean_and_ci

Shift = Callable[[Image.Image, float], Image.Image]


def blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius)) if radius else image


def brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def jpeg(image: Image.Image, quality: float) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def downscale(image: Image.Image, fraction: float) -> Image.Image:
    """Shrink then restore, which is what a low-resolution phone camera does."""

    width, height = image.size
    small = image.resize((max(8, int(width * fraction)), max(8, int(height * fraction))))
    return small.resize((width, height))


#: Each suite starts at its identity level, so "clean" is measured in-run and
#: every cell is comparable to it on the same episodes.
SUITE: dict[str, tuple[Shift, list[float]]] = {
    "blur": (blur, [0.0, 1.0, 2.0, 4.0]),
    "brightness": (brightness, [1.0, 0.6, 0.3, 1.6]),
    "jpeg": (jpeg, [95, 40, 15, 5]),
    "downscale": (downscale, [1.0, 0.5, 0.25, 0.125]),
}


def _load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _predict_all(
    learner: FewShotLearner, paths: list[Path], labels: np.ndarray, indices: np.ndarray, shift: Shift, level: float
) -> dict[str, float]:
    """Accuracy, coverage, set size and OOD rate on shifted queries."""

    correct = covered = flagged = 0
    sizes: list[int] = []
    for index in indices:
        result = learner.predict(shift(_load(paths[index]), level))
        truth = str(labels[index])
        members = result.conformal_set or [result.prediction]
        correct += int(result.prediction == truth)
        covered += int(truth in members)
        sizes.append(len(members))
        flagged += int(result.ood_flag)
    n = len(indices)
    return {"accuracy": correct / n, "coverage": covered / n, "set_size": float(np.mean(sizes)), "ood_rate": flagged / n}


def _balanced(indices: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """The first k of `indices` taken round-robin over classes, not class by class."""

    by_class: dict[str, list[int]] = {}
    for index in indices:
        by_class.setdefault(str(labels[index]), []).append(int(index))
    order: list[int] = []
    for position in range(max(len(v) for v in by_class.values())):
        for members in by_class.values():
            if position < len(members):
                order.append(members[position])
    return np.array(order[:k], dtype=np.int64)


def _recalibrate(
    learner: FewShotLearner, paths: list[Path], labels: np.ndarray, indices: np.ndarray, shift: Shift, level: float, workdir: str
) -> None:
    """Feed shifted, labelled photographs through the library's own correction path."""

    for position, index in enumerate(indices):
        target = os.path.join(workdir, f"insitu_{position}.png")
        shift(_load(paths[index]), level).save(target)
        learner.correct(image_path=target, true_label=str(labels[index]))


def run_cell(
    kind: str, level: float, episodes: list[Episode], paths: list[Path], labels: np.ndarray,
    config: AdaptShotConfig, recalibrate_k: int, workdir: str,
) -> dict[str, Any]:
    shift = SUITE[kind][0]
    before: dict[str, list[float]] = {"accuracy": [], "coverage": [], "set_size": [], "ood_rate": []}
    after: dict[str, list[float]] = {"accuracy": [], "coverage": [], "set_size": []}

    for episode in episodes:
        learner = FewShotLearner(config=config)
        learner.load_support_images(
            [str(paths[i]) for i in episode.support], [str(labels[i]) for i in episode.support]
        )
        measured = _predict_all(learner, paths, labels, episode.query, shift, level)
        for key, value in measured.items():
            before[key].append(value)

        # The mitigation: k shifted, labelled photographs from the episode's
        # calibration split -- images the queries never include -- through
        # correct(). Then the same shifted queries again.
        #
        # Class-balanced, by interleaving. The calibration split is laid out
        # class by class, and taking its first k handed correct() five blurred
        # photographs of two classes and none of the other three: those two
        # prototypes drifted toward "blurry", every blurred query matched them,
        # and coverage fell from 0.66 to 0.40. That was this harness's mistake,
        # and it is the mistake a field user makes by correcting one crop's
        # photos and not another's -- worth knowing, but not what is measured.
        _recalibrate(learner, paths, labels, _balanced(episode.calibration, labels, recalibrate_k), shift, level, workdir)
        again = _predict_all(learner, paths, labels, episode.query, shift, level)
        for key in after:
            after[key].append(again[key])

    def summarise(values: list[float]) -> dict[str, float]:
        mean, half = mean_and_ci(values)
        return {"mean": mean, "ci95_half_width": half}

    return {
        "kind": kind,
        "level": level,
        "identity": level == SUITE[kind][1][0],
        "accuracy": summarise(before["accuracy"]),
        "coverage": summarise(before["coverage"]),
        "set_size": summarise(before["set_size"]),
        "ood_rate": summarise(before["ood_rate"]),
        "after_in_situ_corrections": {
            "k": recalibrate_k,
            "accuracy": summarise(after["accuracy"]),
            "coverage": summarise(after["coverage"]),
            "set_size": summarise(after["set_size"]),
        },
    }


def early_warning(cells: list[dict[str, Any]], target: float) -> dict[str, Any]:
    """Does the OOD flag rate track the coverage that was lost?

    Correlation over the shifted cells between (target - coverage) and the OOD
    rate. High means the flag rises as the guarantee fails; low means it does
    not, and the failure is silent.
    """

    shifted = [c for c in cells if not c["identity"]]
    lost = np.array([max(0.0, target - c["coverage"]["mean"]) for c in shifted])
    flagged = np.array([c["ood_rate"]["mean"] for c in shifted])
    if lost.std() == 0 or flagged.std() == 0:
        return {"correlation": None, "n_cells": len(shifted)}
    return {
        "correlation": float(np.corrcoef(lost, flagged)[0, 1]),
        "n_cells": len(shifted),
        "note": (
            "Pearson correlation across shifted cells between coverage shortfall "
            "(target minus measured, floored at 0) and the OOD flag rate"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/pv_bench"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--recalibrate-k", type=int, default=10, help="shifted labelled photos fed to correct() per episode")
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--output", type=Path, default=Path("results/plantvillage_shift.json"))
    args = parser.parse_args(argv)

    set_deterministic_seed(args.seed)
    config = AdaptShotConfig(backbone=args.backbone, device="cpu", seed=args.seed, conformal_alpha=args.alpha)
    try:
        paths, labels, classes = load_pool(args.data)
    except DatasetMissing as exc:
        print(str(exc))
        return 1
    episodes = sample_episodes(
        labels, classes, n_way=5, k_shot=5, n_calibration=5, n_query=10, episodes=args.episodes, seed=args.seed
    )
    target = 1.0 - args.alpha
    print(f"{len(episodes)} episodes, alpha={args.alpha} (target {target:.0%}), "
          f"{args.recalibrate_k} in-situ corrections per episode\n")
    print(f"{'shift':<11} {'level':>6} {'acc':>6} {'coverage':>15} {'set':>5} {'ood%':>5}  | after k corrections: {'coverage':>15} {'set':>5}")

    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as workdir:
        for kind, (_, levels) in SUITE.items():
            for level in levels:
                cell = run_cell(kind, level, episodes, paths, labels, config, args.recalibrate_k, workdir)
                cells.append(cell)
                c, a = cell["coverage"], cell["after_in_situ_corrections"]["coverage"]
                print(
                    f"{kind:<11} {level:>6} {cell['accuracy']['mean']:>6.2f} "
                    f"{c['mean']:>7.3f} +/- {c['ci95_half_width']:.3f} {cell['set_size']['mean']:>5.2f} "
                    f"{100 * cell['ood_rate']['mean']:>5.1f}  | {a['mean']:>19.3f} +/- {a['ci95_half_width']:.3f} "
                    f"{cell['after_in_situ_corrections']['set_size']['mean']:>5.2f}"
                )

    warning = early_warning(cells, target)
    print(f"\nOOD flag as early warning: correlation with coverage shortfall = "
          f"{warning['correlation'] if warning['correlation'] is None else round(warning['correlation'], 2)}")
    print(f"({time.perf_counter() - started:.0f}s)")

    record = {
        "protocol": {
            "task": "5-way 5-shot", "episodes": args.episodes, "seed": args.seed, "alpha": args.alpha,
            "target_coverage": target, "backbone": args.backbone, "shift_applied_to": "queries only",
            "mitigation": "FewShotLearner.correct() on shifted labelled calibration photographs",
            "recalibrate_k": args.recalibrate_k,
            "suite": {kind: levels for kind, (_, levels) in SUITE.items()},
        },
        "cells": cells,
        "early_warning": warning,
        "dataset": dataset_provenance(args.data),
        "hardware": hardware(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
