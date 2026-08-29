# AdaptShot

**A few-shot image classifier that knows when it doesn't know.** Teach it a new class from a handful of photographs, on an ordinary laptop, with the network switched off. Get back not just a label but a prediction set with a coverage guarantee — and a refusal when it should refuse. Version 0.3.0.

Every page here says, in its first line, who it is for. Pick the door that matches you.

## I have never written code

Start at [**what AdaptShot is, in plain words**](tutorials/00-what-is-this.md) — no installing, ten minutes of reading — and follow the six tutorials in order. They install it, make a prediction on photographs that come with it, teach it your own, read every part of its answer, and correct it. Every command in them is executed by the project's tests on every change; if one does not work for you, that is a bug in the documentation, not in you.

## I have a task

The [**how-to guides**](how-to/run-the-offline-demo.md) are one page per task and assume the tutorials: run the demo offline, check what a machine can do, choose the promise level, save and load, fine-tune, use another backbone, export to ONNX, run the benchmarks, use the Tambua web app, deploy offline, troubleshoot.

## I want to understand why

[**How it works**](understand/how-it-works.md) in one page. [**The guarantee**](understand/the-guarantee.md) — exactly what the prediction set promises, when it holds, and where it stops, with measurements. [**Why CPU-only and offline**](understand/why-cpu-only-and-offline.md). The [algorithm theory](understand/algorithm-theory.md), the [human-in-the-loop design](understand/human-in-the-loop.md), and the [two-page technical note](understand/technical-note.md) with every number traceable.

## I need the exact details

The [**API reference**](reference/api.md), with every public name classified stable or experimental; the [configuration reference](reference/config-reference.md); [errors and warnings](reference/errors.md); [results and artifacts](reference/results-and-artifacts.md) — where every published number comes from; the [Tambua command line](reference/tambua-cli.md); the [changelog](reference/changelog.md).

## I want to contribute

[**Contributing**](contributing/contributing.md) and the [development setup](contributing/development-setup.md) with the five-stage gate; [API stability](contributing/api-stability.md); [how these docs are tested](contributing/how-the-docs-are-tested.md); the [release checklist](contributing/release-checklist.md); the [code of conduct](contributing/code_of_conduct.md).

---

Seven lines, if you would rather just see it:

```python
from adaptshot import FewShotLearner
from adaptshot.data import sample_images

paths, labels = sample_images()                      # twelve real maize-leaf photographs, bundled
learner = FewShotLearner()
learner.load_support_images(paths[:-1], labels[:-1])
result = learner.predict(paths[-1])
print(result.prediction, f"{result.calibrated_confidence:.0%}", result.conformal_set)
```

`pip install adaptshot` first (or `uv add adaptshot`, or `poetry add adaptshot` — it is one package on PyPI, and any of them fetches it). Under five seconds from install to that answer, measured; nothing downloaded after the install.
