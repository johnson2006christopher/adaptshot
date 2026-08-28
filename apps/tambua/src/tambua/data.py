"""Tambua: sample generation and real-image loading.

Two modes:

1. **Placeholder generation** -- deterministic synthetic images, one visually
   distinct pattern per configured class, so the whole loop can be demonstrated
   before anyone has collected a dataset.
2. **Real image loading** -- a folder-per-class directory tree.

The placeholders deliberately look like nothing in particular. An earlier version
drew maize leaves, which was worse in two ways: it hard-coded one domain's
vocabulary into the code, and it invited a viewer to believe the model was
analysing foliage when it was in fact separating drawn shapes. Neutral patterns
make the demo honest about what it is. Real evaluation is #18.
"""

from __future__ import annotations

import colorsys
import hashlib
import os
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Deterministic placeholder generation
# ---------------------------------------------------------------------------

#: How many visually distinct pattern families the generator can produce. Two
#: classes that collide onto the same family are still separable -- they get
#: different hues and densities -- but distinctness degrades past this many.
PATTERN_FAMILIES = 4

IMAGE_SIZE = 224


def _class_signature(class_key: str) -> tuple[int, float, int, int]:
    """Derive stable visual parameters from a class name.

    `hashlib`, not the builtin `hash()`: hash randomisation is seeded per process,
    so the builtin would give a different picture every run and quietly break the
    determinism the project guarantees.

    Returns:
        (family, hue, blob_count, seed) -- fixed for a given class name forever.
    """

    digest = hashlib.blake2b(class_key.encode("utf-8"), digest_size=8).digest()
    family = digest[0] % PATTERN_FAMILIES
    hue = digest[1] / 255.0
    blob_count = 4 + (digest[2] % 9)
    seed = int.from_bytes(digest[4:8], "big")
    return family, hue, blob_count, seed


def _rgb(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def make_placeholder(class_key: str, variant: int = 0, size: int = IMAGE_SIZE) -> Image.Image:
    """Render one placeholder image for a class.

    Args:
        class_key: The class this image is an example of. Fixes the pattern.
        variant: Which example. Jitters position and size so the support set has
            genuine within-class variation rather than N identical copies.
        size: Output edge length in pixels.

    Returns:
        A square RGB image, identical on every run for the same arguments.
    """

    family, hue, blob_count, seed = _class_signature(class_key)
    rng = np.random.default_rng(seed + variant)

    background = _rgb(hue, 0.25, 0.90)
    foreground = _rgb(hue + 0.5, 0.65, 0.55)
    accent = _rgb(hue + 0.5, 0.45, 0.75)

    img = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(img)

    if family == 0:  # scattered discs
        for _ in range(blob_count):
            r = int(rng.integers(size // 14, size // 7))
            cx = int(rng.integers(r, size - r))
            cy = int(rng.integers(r, size - r))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=foreground, outline=accent, width=2)
    elif family == 1:  # diagonal bands
        step = max(6, size // blob_count)
        for offset in range(-size, size * 2, step):
            jitter = int(rng.integers(-3, 4))
            draw.line(
                [(offset + jitter, 0), (offset - size + jitter, size)],
                fill=foreground,
                width=max(3, step // 3),
            )
    elif family == 2:  # concentric rings
        centre = size // 2 + int(rng.integers(-size // 12, size // 12))
        for ring in range(blob_count):
            r = int((ring + 1) * size / (2 * blob_count + 1))
            draw.ellipse(
                [centre - r, centre - r, centre + r, centre + r],
                outline=foreground if ring % 2 else accent,
                width=max(2, size // 40),
            )
    else:  # angular blocks
        for _ in range(blob_count):
            w = int(rng.integers(size // 10, size // 4))
            h = int(rng.integers(size // 10, size // 4))
            x = int(rng.integers(0, size - w))
            y = int(rng.integers(0, size - h))
            draw.rectangle([x, y, x + w, y + h], fill=foreground, outline=accent, width=2)

    return img


def make_unrelated_image(size: int = IMAGE_SIZE, seed: int = 0) -> Image.Image:
    """Render structureless noise, for demonstrating out-of-distribution flagging.

    It belongs to no configured class by construction, which is exactly the
    property the OOD check should notice.
    """

    rng = np.random.default_rng(seed)
    base = rng.integers(80, 180, (size, size, 3), dtype=np.int16)
    noise = rng.integers(-20, 20, (size, size, 3), dtype=np.int16)
    pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


# ---------------------------------------------------------------------------
# Sample dataset generation
# ---------------------------------------------------------------------------


def generate_samples(
    output_dir: str,
    class_keys: Sequence[str],
    n_support: int = 5,
    n_query: int = 3,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Write a placeholder dataset covering the given classes.

    The classes come from the caller -- in practice, from the loaded config -- so
    the generated label set always matches what the application was configured to
    recognise. Nothing here knows what the classes mean.

    Args:
        output_dir: Directory to write PNGs into. Created if absent.
        class_keys: The classes to generate examples for.
        n_support: Images per class for the support set.
        n_query: Images per class for the query set. 0 to skip.

    Returns:
        (support_paths, support_labels, query_paths, query_labels)

    Raises:
        ValueError: If `class_keys` is empty.
    """

    if not class_keys:
        raise ValueError(
            "generate_samples needs at least one class; the loaded config defines none"
        )

    os.makedirs(output_dir, exist_ok=True)

    support_paths: list[str] = []
    support_labels: list[str] = []
    query_paths: list[str] = []
    query_labels: list[str] = []

    for class_key in class_keys:
        for i in range(n_support):
            path = os.path.join(output_dir, f"{class_key}_support_{i:02d}.png")
            make_placeholder(class_key, variant=i).save(path)
            support_paths.append(path)
            support_labels.append(class_key)
        for i in range(n_query):
            # Offset the variant so query images are never byte-identical to a
            # support image -- otherwise the demo measures memorisation.
            path = os.path.join(output_dir, f"{class_key}_query_{i:02d}.png")
            make_placeholder(class_key, variant=1000 + i).save(path)
            query_paths.append(path)
            query_labels.append(class_key)

    return support_paths, support_labels, query_paths, query_labels


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
