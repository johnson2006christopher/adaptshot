# Studio Moved to Its Own Repository

`src/adaptshot/studio/` was **1,822 lines — 23% of the library — covered by four
tests**. It was a Gradio desktop application, not few-shot learning and not a
library. It has been extracted ([#21]).

## Why

A reviewer measuring this project was measuring a quarter of it as GUI code,
while the argument the project actually makes — few-shot classification with a
coverage guarantee, on a CPU — is carried by a much smaller subset. The GUI also
dragged `gradio`, `pandas`, `onnx` and `onnxruntime` into the project's identity
and its type checking, and a GUI has a different release cadence from a library:
coupling them made every interface tweak a library release.

## Nothing was lost

The full commit history was extracted before anything was deleted and lives on
the [`studio-extract`](https://github.com/johnson2006christopher/adaptshot/tree/studio-extract)
branch — thirteen commits, `__init__.py`, `app.py` and `utils.py` at the root,
ready to become a repository of its own.

## What to use instead

**[Tambua](tambua-complete-guide.md)** is the maintained application. It is a
separate distribution built on AdaptShot, which is what keeps the library a
library:

```bash
pip install tambua
tambua
```

Its domain comes from a configuration file, so the same code is a crop-disease
tool or a solar-panel inspector depending on which config is loaded.

## For your own code

There is no drop-in replacement inside the library, by design. Build on the
public API:

```python
from adaptshot import FewShotLearner

learner = FewShotLearner()
learner.load_support_images(paths, labels)
result = learner.predict("photo.jpg")
```

The `gui` extra and the `adaptshot-studio` console script are gone.

[#21]: https://github.com/johnson2006christopher/adaptshot/issues/21
