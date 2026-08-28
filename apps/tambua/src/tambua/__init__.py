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
    engine.load_images_from_dir("my_photos/")   # one folder per class
    result = engine.identify("photo.jpg")
    print(result.local_name, result.confidence, result.action)

.. note::

   Tambua ships no images. It used to generate them -- drawn shapes offered as
   "sample data" -- and that was removed in #53: a number measured on drawn
   patterns is not a result. Five real photographs per class is the input.
   :func:`tambua.data.inspect_folder` reports whether a folder can support
   training before a run is spent finding out.
"""

from tambua import data
from tambua.config import ClassInfo, TambuaConfig, load_config
from tambua.engine import (
    DEFAULT_CONFIG,
    Identification,
    TambuaEngine,
    bundled_config,
    combined_action,
)

__all__ = [
    "DEFAULT_CONFIG",
    "ClassInfo",
    "Identification",
    "TambuaConfig",
    "TambuaEngine",
    "bundled_config",
    "combined_action",
    "data",
    "load_config",
]
