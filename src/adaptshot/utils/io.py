"""I/O utilities for path validation, serialization, and safe data conversion."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union, cast

import numpy as np
import torch

# Module logger (configure at app level; defaults to WARNING if unconfigured)
logger = logging.getLogger(__name__)


def validate_path(path: Union[str, Path], must_exist: bool = False, is_dir: bool = False) -> Path:
    """
    Validate and normalize a file or directory path.

    Prevents silent failures from typos, missing directories, or incorrect path types.
    Used extensively in benchmark saving, checkpointing, and dataset loading.

    Args:
        path: Path string or Path object
        must_exist: If True, raises FileNotFoundError if path doesn't exist
        is_dir: If True, treats as directory (creates it if missing when must_exist=False)

    Returns:
        Path: Normalized, absolute pathlib.Path object
    """
    path_obj = Path(path).resolve()

    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path_obj}")

    if is_dir:
        if path_obj.exists() and not path_obj.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path_obj}")
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: %s", path_obj)

    return path_obj


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file with pretty formatting.

    Automatically creates parent directories if missing. Uses UTF-8 encoding
    to safely handle non-ASCII characters in logs or metadata.

    Args:
        data: Dictionary to serialize
        path: Target file path
        indent: JSON indentation level (default: 2)
    """
    path_obj = validate_path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.info("Saved JSON to %s", path_obj)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        path: Source file path (must exist)

    Returns:
        Dict containing parsed JSON data
    """
    path_obj = validate_path(path, must_exist=True)

    with open(path_obj, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch.Tensor to numpy array, handling device and gradients safely.

    Essential for bridging PyTorch inference outputs with NumPy/FAISS pipelines
    without triggering gradient tracking errors or CUDA-to-CPU transfer bugs.

    Args:
        tensor: Input torch.Tensor (any shape, any device)

    Returns:
        np.ndarray: Equivalent numpy array pinned to CPU memory
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()