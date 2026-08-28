"""Tambua — few-shot image identification with a human in the loop.

*Tambua* is Swahili for "identify". The application learns a classifier from a
handful of labelled examples per class and serves it through a Gradio web UI.

The domain comes from a configuration file, not from the code. Loaded with the
bundled ``maize.yaml`` it is MziziGuard, a crop-disease tool that speaks Swahili;
loaded with ``solar_panel.yaml`` it triages photovoltaic modules for an off-grid
technician. Same code, same loop, different vocabulary.

Usage::

    tambua                                   # the flagship config
    tambua --config path/to/your/domain.yaml

    # or, programmatically:
    from tambua import TambuaEngine, bundled_config

    engine = TambuaEngine(bundled_config("solar_panel"))
    engine.initialize_with_samples(n_support=5)
    result = engine.identify("photo.jpg")
    print(result.local_name, result.confidence, result.action)

.. warning::

   The bundled sample images are generated procedurally by :mod:`tambua.data`.
   They are drawn patterns, not photographs, and exist so the loop can be
   demonstrated before a dataset exists. Evaluation on real data is issue #18.
"""

from tambua import data
from tambua.config import ClassInfo, TambuaConfig, load_config
from tambua.engine import DEFAULT_CONFIG, Identification, TambuaEngine, bundled_config

__all__ = [
    "DEFAULT_CONFIG",
    "ClassInfo",
    "Identification",
    "TambuaConfig",
    "TambuaEngine",
    "bundled_config",
    "data",
    "load_config",
]
