"""MziziGuard: Data loading and sample generation utilities.

Supports two modes:
  1. **Sample generation** — synthetic leaf images for quick demos (zero dependencies).
  2. **Real image loading** — load images from a folder-per-class directory tree.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Colour palettes for synthetic leaf generation
# ---------------------------------------------------------------------------

LEAF_GREEN = (34, 139, 34)
LEAF_GREEN_LIGHT = (60, 179, 113)
BLIGHT_BROWN = (139, 69, 19)
BLIGHT_TAN = (210, 180, 140)
SPOT_GRAY = (128, 128, 128)
SPOT_DARK = (80, 80, 80)
SOIL_BG = (160, 140, 100)


def _draw_vein(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    draw.line([(x0, y0), (x1, y1)], fill=(0, 100, 0), width=1)


# ---------------------------------------------------------------------------
# Synthetic leaf generators
# ---------------------------------------------------------------------------


def make_healthy_leaf(size: int = 224) -> Image.Image:
    """Generate a healthy green maize leaf with veins."""
    img = Image.new("RGB", (size, size), SOIL_BG)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    leaf_w, leaf_h = size // 4, size // 3
    draw.ellipse(
        [(cx - leaf_w, cy - leaf_h), (cx + leaf_w, cy + leaf_h)],
        fill=LEAF_GREEN,
        outline=LEAF_GREEN_LIGHT,
        width=2,
    )
    midrib_top = (cx, cy - leaf_h + 10)
    midrib_bot = (cx, cy + leaf_h - 10)
    draw.line([midrib_top, midrib_bot], fill=(0, 80, 0), width=3)
    for i in range(-3, 4):
        if i == 0:
            continue
        y = cy + i * (leaf_h // 4)
        offset = abs(i) * 8
        _draw_vein(draw, cx, y, cx - leaf_w + 20 + offset, y - 5)
        _draw_vein(draw, cx, y, cx + leaf_w - 20 - offset, y + 5)
    return img


def make_blight_leaf(size: int = 224) -> Image.Image:
    """Generate a maize leaf with Northern Leaf Blight lesions."""
    img = make_healthy_leaf(size)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    leaf_h = size // 3
    for _ in range(random.randint(3, 6)):
        lx = cx + random.randint(-size // 5, size // 5)
        ly = cy + random.randint(-leaf_h + 30, leaf_h - 30)
        lw = random.randint(8, 18)
        lh = random.randint(25, 55)
        draw.ellipse(
            [(lx - lw, ly - lh), (lx + lw, ly + lh)],
            fill=BLIGHT_TAN,
            outline=BLIGHT_BROWN,
            width=2,
        )
        draw.ellipse(
            [(lx - lw // 2, ly - lh // 3), (lx + lw // 2, ly + lh // 3)],
            fill=BLIGHT_BROWN,
        )
    return img


def make_gray_leaf_spot(size: int = 224) -> Image.Image:
    """Generate a maize leaf with Gray Leaf Spot lesions."""
    img = make_healthy_leaf(size)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    leaf_h = size // 3
    for _ in range(random.randint(4, 8)):
        sx = cx + random.randint(-size // 5, size // 5)
        sy = cy + random.randint(-leaf_h + 30, leaf_h - 30)
        sw = random.randint(6, 14)
        sh = random.randint(10, 25)
        draw.rectangle(
            [(sx - sw, sy - sh), (sx + sw, sy + sh)],
            fill=SPOT_GRAY,
            outline=SPOT_DARK,
            width=2,
        )
    return img


def make_non_leaf(size: int = 224) -> Image.Image:
    """Generate a non-leaf image (soil/noise) for OOD demo."""
    img = np.random.randint(80, 180, (size, size, 3), dtype=np.uint8)
    noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


# Map disease names to generators (synthetic mode)
DISEASE_GENERATORS: dict[str, Callable[[], Image.Image]] = {
    "healthy_maize": make_healthy_leaf,
    "northern_leaf_blight": make_blight_leaf,
    "gray_leaf_spot": make_gray_leaf_spot,
}


# ---------------------------------------------------------------------------
# Sample dataset generation
# ---------------------------------------------------------------------------


def generate_samples(
    output_dir: str,
    n_support: int = 5,
    n_query: int = 3,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Generate synthetic leaf images for training and testing.

    Returns:
        (support_paths, support_labels, query_paths, query_labels)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    support_paths: list[str] = []
    support_labels: list[str] = []
    query_paths: list[str] = []
    query_labels: list[str] = []

    for disease_name, generator in DISEASE_GENERATORS.items():
        for i in range(n_support):
            path = os.path.join(output_dir, f"{disease_name}_support_{i:02d}.png")
            img = generator()
            img.save(path)
            support_paths.append(path)
            support_labels.append(disease_name)
        for i in range(n_query):
            path = os.path.join(output_dir, f"{disease_name}_query_{i:02d}.png")
            img = generator()
            img.save(path)
            query_paths.append(path)
            query_labels.append(disease_name)

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
