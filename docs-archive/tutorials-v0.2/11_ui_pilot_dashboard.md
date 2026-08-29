# 11. The Interface Moved Out of the Library

This chapter used to document `src/adaptshot/ui/app.py`, a Gradio dashboard that
shipped inside AdaptShot. **It has been removed** ([#22]).

## Why

The library shipped *two* Gradio interfaces — `adaptshot.ui.app` and
`adaptshot.studio` — which is one more than the maximum. Neither was the
application anyone was pointed at, both had to be maintained, and both pulled a
web framework into a library whose entire premise is that it runs on a CPU in a
field with no internet.

Nothing was lost. Every capability in `adaptshot.ui.app` — loading a support set,
predicting on a query, routing a correction back into the learner — exists in
both `adaptshot.studio` and in Tambua, in more complete form.

## Where the interface lives now

**[Tambua](../guides/tambua-complete-guide.md)** is the application. It is a
separate distribution built on AdaptShot, so the library stays a library:

```bash
pip install tambua
tambua
```

Its domain comes from a configuration file. Loaded with `maize.yaml` it is
MziziGuard; with `solar_panel.yaml` it is SolarCheck. See the
[complete guide](../guides/tambua-complete-guide.md).

!!! warning "MziziGuard has never been deployed"

    Its sample images are generated procedurally — drawn patterns, not
    photographs — so no accuracy figure from them means anything about maize.
    Evaluation on real data is [#18]. Removing the generated images entirely,
    in favour of real photographs only, is [#53].

[#18]: https://github.com/johnson2006christopher/adaptshot/issues/18
[#53]: https://github.com/johnson2006christopher/adaptshot/issues/53

`adaptshot.studio` remains for now and is being extracted to its own repository
([#21]).

## What this means for your code

If you were importing from `adaptshot.ui`, there is no drop-in replacement inside
the library, by design:

```python
# No longer available
from adaptshot.ui.app import build_ui

# Build on the library directly
from adaptshot import FewShotLearner
learner = FewShotLearner()
learner.load_support_images(paths, labels)
result = learner.predict("photo.jpg")
```

The `adaptshot[ui]` extra is gone. `pip install tambua` is what you want.

[#21]: https://github.com/johnson2006christopher/adaptshot/issues/21
[#22]: https://github.com/johnson2006christopher/adaptshot/issues/22
