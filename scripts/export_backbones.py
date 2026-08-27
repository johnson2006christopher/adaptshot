#!/usr/bin/env python3
"""Export pretrained backbone models to ONNX format for torch-free inference.

Usage:
    python scripts/export_backbones.py --all
    python scripts/export_backbones.py --backbone resnet18

Generated models are saved to src/adaptshot/data/ for package bundling.
Requires: torch, torchvision, onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def export_backbone(
    backbone_name: str,
    output_dir: str = "src/adaptshot/data",
    opset_version: int = 17,
) -> Dict[str, Any]:
    """Export a pretrained backbone to ONNX format.

    Args:
        backbone_name: Name of backbone ("resnet18" or "mobilenet_v3_small").
        output_dir: Directory to save ONNX files.
        opset_version: ONNX opset version.

    Returns:
        Dict with export metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pretrained backbone
    if backbone_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Identity()
        input_shape = (1, 3, 224, 224)
    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        model.classifier = nn.Identity()
        input_shape = (1, 3, 224, 224)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    onnx_path = output_path / f"{backbone_name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )

    # Compute checksum
    with open(onnx_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    # Save metadata
    metadata = {
        "backbone": backbone_name,
        "opset_version": opset_version,
        "input_shape": list(input_shape),
        "sha256": sha256,
        "file_size_mb": round(file_size_mb, 2),
        "pretrained_weights": "IMAGENET1K_V1",
    }

    meta_path = output_path / f"{backbone_name}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def verify_export(backbone_name: str, output_dir: str = "src/adaptshot/data") -> bool:
    """Verify that the exported ONNX model is loadable and outputs correct shape."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed; cannot verify")
        return True

    onnx_path = Path(output_dir) / f"{backbone_name}.onnx"
    if not onnx_path.exists():
        print(f"  [FAIL] ONNX file not found: {onnx_path}")
        return False

    session = ort.InferenceSession(str(onnx_path))
    dummy: np.ndarray = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {"input": dummy})
    embedding: np.ndarray = outputs[0]

    expected_dim = {"resnet18": 512, "mobilenet_v3_small": 576}[backbone_name]
    if embedding.shape[1] != expected_dim:
        print(f"  [FAIL] Expected dim {expected_dim}, got {embedding.shape[1]}")
        return False

    print(f"  [PASS] Output shape: {embedding.shape}, dim={embedding.shape[1]}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AdaptShot backbones to ONNX")
    parser.add_argument(
        "--backbone",
        type=str,
        help="Specific backbone to export",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all available backbones",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/adaptshot/data",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported models with onnxruntime",
    )
    args = parser.parse_args()

    backbones = []
    if args.backbone:
        backbones = [args.backbone]
    elif args.all:
        backbones = ["resnet18", "mobilenet_v3_small"]
    else:
        parser.print_help()
        return 1

    print(f"Exporting backbones to {args.output_dir}/")
    for name in backbones:
        print(f"\n📦 {name}...")
        metadata = export_backbone(name, output_dir=args.output_dir)
        print(f"  Exported: {name}.onnx ({metadata['file_size_mb']} MB)")
        print(f"  SHA-256: {metadata['sha256'][:16]}...")

        if args.verify:
            verify_export(name, output_dir=args.output_dir)

    print("\n✅ Export complete.")
    print(f"Models saved to: {Path(args.output_dir).resolve()}")
    print("Update pyproject.toml [tool.setuptools.package-data] if needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
