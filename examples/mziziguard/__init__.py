"""MziziGuard — Crop disease detection for smallholder farmers.

A working application powered by AdaptShot few-shot learning.
Provides a Gradio web UI for diagnosing crop diseases from photos.

Usage::

    python -m examples.mziziguard.app

    # or, programmatically:
    from examples.mziziguard.engine import MziziGuard

    guard = MziziGuard()
    guard.initialize_with_samples(n_support=5)
    result = guard.diagnose("photo.jpg")
    print(result.swahili, result.confidence)
"""

from .engine import MziziGuard, DiagnosisResult, DiseaseInfo
from . import data

__all__ = ["MziziGuard", "DiagnosisResult", "DiseaseInfo", "data"]
