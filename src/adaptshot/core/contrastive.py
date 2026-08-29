"""Moved to ``adaptshot.training.contrastive`` in 0.3.0.

It trains a projection head by gradient descent, which is what ``training/``
is for; ``core/`` was where it happened to be written. The public names are
unchanged and still importable from the top level as ``adaptshot.ContrastiveConfig``
and ``adaptshot.ContrastivePrototypeLearner``, which is the supported way.

This module re-exports them so that existing imports keep working, and warns so
that they get updated. It is the first use of the deprecation policy in
CONTRIBUTING.md: the alias stays for one minor release and is removed in 0.4.0.
"""

from __future__ import annotations

import warnings

from ..training.contrastive import ContrastiveConfig, ContrastivePrototypeLearner

warnings.warn(
    "adaptshot.core.contrastive moved to adaptshot.training.contrastive in 0.3.0; "
    "this alias will be removed in 0.4.0. Import from adaptshot or "
    "adaptshot.training.contrastive instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ContrastiveConfig", "ContrastivePrototypeLearner"]
