#!/usr/bin/env python3
"""Export AdaptShot backbone models to ONNX format for lightweight inference.

Run this once (requires torch + torchvision installed) to generate the .onnx
files bundled with the package. These enable torch-free inference via ONNX Runtime.

Usage:
    pip install torch torchvision
    python scripts/export_backbones.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "adaptshot" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONES = {
    "resnet18": lambda: models.resnet18(weights="IMAGENET1K_V1"),
    "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights="IMAGENET1K_V1"),
}


def export_backbone(name: str, factory) -> None:
    """Export a single backbone to ONNX format."""
    model = factory()
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = OUTPUT_DIR / f"{name}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  EXPORTED  {name}.onnx  ({size_mb:.1f} MB)")


def main() -> None:
    print("Exporting AdaptShot backbones to ONNX format...\n")
    for name, factory in BACKBONES.items():
        export_backbone(name, factory)
    print(f"\nDone – models saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
