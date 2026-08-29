"""Tambua: loading real images, and telling you when a folder is not usable.

Tambua ships no images. It used to generate them -- coloured shapes drawn with
`ImageDraw`, offered through the interface as "sample data" -- and that was
removed in #53. Drawn patterns are not data. A model that separates a blue ring
from a green blob at 95% has told you nothing about maize, and anyone quoting
that number will be right to be disbelieved.

The premise is few-shot: five photographs per class. Anyone who wants this has
photographs, which is why they want it. So what ships is a path to their data
and an honest account of whether it is usable, not a substitute for it.

The generator still exists, under `tests/support/images.py`, where deterministic
licence-free images are the right tool for asserting that the pipeline runs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

#: Below this, a class contributes almost nothing to a prototype and the result
#: is dominated by whichever example happened to be photographed well.
MIN_IMAGES_PER_CLASS = 3

#: Backbones resize to 224x224. An image smaller than this is being upscaled,
#: which invents detail the camera never captured.
MIN_EDGE_PIXELS = 224


@dataclass(frozen=True)
class FolderProblem:
    """Something wrong with a training folder, and how to fix it."""

    where: str
    problem: str
    remedy: str


def describe_expected_layout(class_keys: Sequence[str]) -> str:
    """The folder layout this configuration expects, as text for the interface."""

    shown = list(class_keys)[:3]
    lines = [f"    {key}/" for key in shown]
    if len(class_keys) > len(shown):
        lines.append(f"    ...and {len(class_keys) - len(shown)} more")
    return "your_photos/\n" + "\n".join(f"{line}\n        photo_01.jpg" for line in lines)


def inspect_folder(image_dir: str, class_keys: Sequence[str]) -> list[FolderProblem]:
    """Report everything wrong with a training folder, before training on it.

    Finding out that half a class is unreadable *after* a model is trained wastes
    the training and hides the cause. Every problem is reported at once, naming
    the folder and the fix, in the same style as the config validator.

    Args:
        image_dir: Root directory, one subdirectory per class.
        class_keys: The classes the loaded configuration defines.

    Returns:
        Problems found, in reading order. Empty means the folder is usable.
    """

    problems: list[FolderProblem] = []

    if not os.path.isdir(image_dir):
        return [
            FolderProblem(
                where=image_dir,
                problem="is not a directory",
                remedy="point at a folder containing one subfolder per class",
            )
        ]

    subdirs = sorted(
        entry for entry in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, entry))
    )
    if not subdirs:
        return [
            FolderProblem(
                where=image_dir,
                problem="contains no subfolders",
                remedy=(
                    "create one subfolder per class, named exactly as in the "
                    "config: " + ", ".join(class_keys)
                ),
            )
        ]

    configured = set(class_keys)
    for name in subdirs:
        if name not in configured:
            problems.append(
                FolderProblem(
                    where=os.path.join(image_dir, name),
                    problem=f'"{name}" is not a class in the loaded configuration',
                    remedy=(
                        "rename it to one of: " + ", ".join(sorted(configured))
                        + ", or add it to the config under a domain's classes:"
                    ),
                )
            )
            continue

        folder = os.path.join(image_dir, name)
        usable = 0
        for filename in sorted(os.listdir(folder)):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    img.verify()
            except (UnidentifiedImageError, OSError):
                problems.append(
                    FolderProblem(
                        where=path,
                        problem="could not be read as an image",
                        remedy="remove it, or re-export it as JPEG or PNG",
                    )
                )
                continue
            if min(width, height) < MIN_EDGE_PIXELS:
                problems.append(
                    FolderProblem(
                        where=path,
                        problem=f"is {width}x{height}, smaller than {MIN_EDGE_PIXELS}px",
                        remedy=(
                            "use the original photograph; upscaling invents detail "
                            "the camera never captured"
                        ),
                    )
                )
                continue
            usable += 1

        if usable < MIN_IMAGES_PER_CLASS:
            problems.append(
                FolderProblem(
                    where=folder,
                    problem=f"has {usable} usable image{'' if usable == 1 else 's'}",
                    remedy=(
                        f"at least {MIN_IMAGES_PER_CLASS} are needed; five or more "
                        "gives the prototype something to average"
                    ),
                )
            )

    missing = configured - set(subdirs)
    for name in sorted(missing):
        problems.append(
            FolderProblem(
                where=os.path.join(image_dir, name),
                problem=f'no folder for configured class "{name}"',
                remedy=(
                    "add photographs for it, or remove the class from the config "
                    "-- a class with no examples can never be predicted"
                ),
            )
        )

    return problems


def render_problems(problems: Sequence[FolderProblem]) -> str:
    """Format folder problems for a person, not a log."""

    if not problems:
        return ""
    head = f"{len(problems)} problem{'s' if len(problems) != 1 else ''} with this folder:"
    body = "\n".join(f"\n{p.where}: {p.problem}\n  {p.remedy}" for p in problems)
    return head + "\n" + body


# ---------------------------------------------------------------------------
# Real image loading (ImageFolder-style)
# ---------------------------------------------------------------------------


def list_classes_from_dir(root_dir: str) -> list[str]:
    """Discover class names from subdirectory names."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    classes = []
    for name in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and not name.startswith("."):
            classes.append(name)
    if not classes:
        raise ValueError(f"No class directories found in {root_dir}")
    return classes


def scan_image_extensions() -> list[str]:
    """Return supported image file extensions."""
    return [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]


def load_from_folders(
    root_dir: str,
    max_per_class: int = 0,
) -> tuple[list[str], list[str]]:
    """Load images organized in subdirectories (one folder per class).

    Args:
        root_dir: Directory containing one subdirectory per class.
        max_per_class: If > 0, limit images per class. 0 = all.

    Returns:
        (image_paths, labels) suitable for FewShotLearner.load_support_images().
    """
    paths: list[str] = []
    labels: list[str] = []
    extensions = {ext.lower() for ext in scan_image_extensions()}

    for class_name in list_classes_from_dir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        count = 0
        for fname in sorted(os.listdir(class_dir)):
            if not any(fname.lower().endswith(ext) for ext in extensions):
                continue
            paths.append(os.path.join(class_dir, fname))
            labels.append(class_name)
            count += 1
            if max_per_class > 0 and count >= max_per_class:
                break

    if not paths:
        raise ValueError(
            f"No images found in {root_dir}. Organize images as: "
            f"root/class_name/*.png"
        )
    return paths, labels
