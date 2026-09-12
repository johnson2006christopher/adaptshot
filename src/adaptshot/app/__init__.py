"""Tambua — few-shot image identification with a human in the loop.

*Tambua* is Swahili for "identify". The application learns a classifier from a
handful of labelled examples per class and serves it through a Gradio web UI.

Since v0.3.1 it ships inside the library as an optional extra::

    pip install "adaptshot[app]"

The domain comes from a configuration file, not from the code. Loaded with the
bundled ``maize.yaml`` it is MziziGuard, a crop-disease tool that speaks Swahili;
loaded with ``solar_panel.yaml`` it triages photovoltaic modules for an off-grid
technician. Same code, same loop, different vocabulary.

Usage::

    tambua                                   # the flagship config
    tambua --config path/to/your/domain.yaml

    # or, programmatically:
    from adaptshot.app import TambuaEngine, bundled_config

    engine = TambuaEngine(bundled_config("solar_panel"))
    engine.load_images_from_dir("my_photos/")   # one folder per class
    result = engine.identify("photo.jpg")
    print(result.local_name, result.confidence, result.action)

This module and everything it imports stay gradio-free: the engine, the config
loader and the data helpers work on a core install. Only
:mod:`adaptshot.app.ui` imports gradio, and only the ``tambua`` command reaches
for it — so ``import adaptshot.app`` costs a core user nothing.

.. note::

   Tambua ships no images. It used to generate them -- drawn shapes offered as
   "sample data" -- and that was removed in #53: a number measured on drawn
   patterns is not a result. Five real photographs per class is the input.
   :func:`adaptshot.app.data.inspect_folder` reports whether a folder can
   support training before a run is spent finding out.
"""

from adaptshot.app import data
from adaptshot.app.config import ClassInfo, TambuaConfig, load_config
from adaptshot.app.engine import (
    DEFAULT_CONFIG,
    Identification,
    TambuaEngine,
    bundled_config,
    bundled_configs,
    combined_action,
)

__all__ = [
    "DEFAULT_CONFIG",
    "ClassInfo",
    "Identification",
    "TambuaConfig",
    "TambuaEngine",
    "bundled_config",
    "bundled_configs",
    "combined_action",
    "data",
    "load_config",
]
