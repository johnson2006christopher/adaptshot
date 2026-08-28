"""Tambua — few-shot image classification with a human in the loop.

*Tambua* is Swahili for "identify". The application learns a classifier from a
handful of labelled examples per class and serves it through a Gradio web UI.

The domain comes from a configuration file, not from the code. Loaded with
``configs/maize.yaml`` it is MziziGuard, a crop-disease tool that speaks Swahili;
loaded with another config it classifies something else entirely.

Usage::

    tambua --config configs/maize.yaml

    # or, programmatically:
    from tambua.engine import MziziGuard

    guard = MziziGuard()
    guard.initialize_with_samples(n_support=5)
    result = guard.diagnose("photo.jpg")
    print(result.swahili, result.confidence)

.. warning::

   The bundled sample images are generated procedurally by :mod:`tambua.data`.
   They are drawn shapes, not photographs. See issue #18 for evaluation on real
   data.
"""

from tambua import data
from tambua.engine import DiagnosisResult, DiseaseInfo, MziziGuard

__all__ = ["DiagnosisResult", "DiseaseInfo", "MziziGuard", "data"]
