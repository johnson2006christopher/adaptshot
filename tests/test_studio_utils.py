"""Tests for AdaptShot Studio helper functions."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.adaptshot.studio.utils import collect_image_sources, discover_images_in_folder


def test_discover_images_in_folder_recursively(tmp_path: Path) -> None:
    """Ensure folder discovery finds nested image files."""

    root = tmp_path / "dataset"
    class_a = root / "class_a"
    nested = class_a / "nested"
    nested.mkdir(parents=True)
    image_a = class_a / "a.png"
    image_b = nested / "b.jpg"
    Image.new("RGB", (16, 16), color="red").save(image_a)
    Image.new("RGB", (16, 16), color="blue").save(image_b)

    discovered = discover_images_in_folder(root)

    assert discovered == sorted([image_a, image_b])


def test_collect_image_sources_deduplicates_file_and_folder(tmp_path: Path) -> None:
    """Ensure mixed uploads and folder imports stay unique and sorted."""

    root = tmp_path / "dataset"
    root.mkdir()
    image_a = root / "a.png"
    image_b = root / "b.png"
    Image.new("RGB", (16, 16), color="red").save(image_a)
    Image.new("RGB", (16, 16), color="blue").save(image_b)

    collected = collect_image_sources([image_a], folder_text=str(root))

    assert collected == sorted([image_a.resolve(), image_b.resolve()])
