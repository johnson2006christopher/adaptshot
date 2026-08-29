"""Episode sampling and embedding for the PlantVillage benchmark (#18, #19).

This lives in `benchmarks/` rather than in the package, deliberately.
PlantVillage is not a dependency of AdaptShot and must not become one: the
library ships without it, and a user who never benchmarks never downloads it.

The download is a separate, manual, documented step::

    python scripts/fetch_plantvillage.py --out data/pv_bench \\
        --per-class 20 --preset benchmark

Nothing here reaches the network. If the data is absent the caller is told to
run that command; it is never fetched silently.

**Every method sees the same embeddings and the same episodes.** Embeddings are
computed once for the whole pool and indexed into per episode, so a difference
between two methods cannot come from having seen different pixels, and a
baseline cannot be handicapped by a different sampling draw (#19).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adaptshot.config.settings import AdaptShotConfig
from adaptshot.core.extractor import extract_embedding

#: Extensions PlantVillage actually uses. Anything else in the directory is not
#: an image we downloaded and is skipped rather than guessed at.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"})


class DatasetMissing(RuntimeError):
    """Raised when the benchmark data has not been downloaded."""


@dataclass(frozen=True)
class Episode:
    """One n-way k-shot problem, as indices into the embedded pool.

    Indices rather than arrays so that every method is handed the identical
    rows, and so that an episode is small enough to record in the results file
    alongside the number it produced.
    """

    classes: tuple[str, ...]
    support: np.ndarray
    calibration: np.ndarray
    query: np.ndarray

    def labels_for(self, indices: np.ndarray, pool_labels: np.ndarray) -> np.ndarray:
        return pool_labels[indices]


def load_pool(root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """Return (paths, labels, class names) for every downloaded image.

    Sorted at every level, so the pool is identical on any machine that ran the
    same fetch command -- directory iteration order is not.
    """

    if not root.is_dir():
        raise DatasetMissing(
            f"{root} does not exist. Download the benchmark pool first:\n"
            f"  python scripts/fetch_plantvillage.py --out {root} "
            "--per-class 20 --preset benchmark"
        )

    classes = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    if not classes:
        raise DatasetMissing(f"{root} contains no class directories")

    paths: list[Path] = []
    labels: list[str] = []
    for class_name in classes:
        images = sorted(
            path
            for path in (root / class_name).iterdir()
            if path.suffix in IMAGE_SUFFIXES
        )
        paths.extend(images)
        labels.extend([class_name] * len(images))

    return paths, np.array(labels, dtype=object), classes


def embed_pool(
    paths: list[Path],
    config: AdaptShotConfig,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Embed every image once. Cached on disk, keyed by backbone and pool size.

    Embedding dominates the runtime -- 100 episodes touch the same images
    repeatedly -- and recomputing per episode would make the sweep an hour
    instead of a minute, while producing identical numbers.
    """

    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            cached.shape[0] == len(paths)
            and str(cached.dtype) == "float32"
        ):
            return cached

    embeddings = np.stack(
        [np.asarray(extract_embedding(str(path), config), dtype=np.float32) for path in paths]
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
    return embeddings


def sample_episodes(
    labels: np.ndarray,
    classes: list[str],
    *,
    n_way: int,
    k_shot: int,
    n_calibration: int,
    n_query: int,
    episodes: int,
    seed: int,
) -> list[Episode]:
    """Draw `episodes` disjoint-split n-way k-shot problems.

    Support, calibration and query are disjoint within an episode. The
    calibration split exists because split-conformal's guarantee is only a
    guarantee when the scores it calibrates on are held out from the queries it
    then covers -- calibrating on the query set would report a coverage number
    that means nothing.

    There is no train/test *class* split, because AdaptShot never trains the
    backbone: it is frozen ImageNet, so every PlantVillage class is equally
    novel to it and there is no leakage to prevent.
    """

    needed = k_shot + n_calibration + n_query
    if len(classes) < n_way:
        raise ValueError(f"{n_way}-way needs {n_way} classes, the pool has {len(classes)}")

    indices_by_class = {name: np.flatnonzero(labels == name) for name in classes}
    usable = [name for name in classes if len(indices_by_class[name]) >= needed]
    if len(usable) < n_way:
        raise ValueError(
            f"{n_way}-way {k_shot}-shot with {n_calibration} calibration and "
            f"{n_query} query images needs {needed} images per class; only "
            f"{len(usable)} of {len(classes)} classes have that many"
        )

    rng = np.random.default_rng(seed)
    drawn: list[Episode] = []
    for _ in range(episodes):
        chosen = rng.choice(len(usable), size=n_way, replace=False)
        support, calibration, query = [], [], []
        for position in chosen:
            pool = indices_by_class[usable[position]]
            picked = rng.choice(pool, size=needed, replace=False)
            support.extend(picked[:k_shot])
            calibration.extend(picked[k_shot : k_shot + n_calibration])
            query.extend(picked[k_shot + n_calibration :])
        drawn.append(
            Episode(
                classes=tuple(usable[position] for position in chosen),
                support=np.array(support, dtype=np.int64),
                calibration=np.array(calibration, dtype=np.int64),
                query=np.array(query, dtype=np.int64),
            )
        )
    return drawn


def dataset_provenance(root: Path) -> dict[str, Any]:
    """Whatever the fetch script recorded, so results name the exact bytes."""

    manifest = root / "manifest.json"
    if not manifest.is_file():
        return {"manifest": "absent"}
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "repo": record.get("repo"),
        "commit": record.get("commit"),
        "licence": record.get("licence"),
        "citation": record.get("citation"),
        "preset": record.get("preset"),
        "n_files": len(record.get("files", {})),
    }
