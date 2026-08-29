# Where to go next

> **For:** someone who has finished the five tutorials and now has a task in mind. Two minutes.

You can install AdaptShot, teach it from a folder of photographs, read every part of its answer, correct it, and save it. Everything from here on is organised by what you want to do, not by what there is to know.

## I have a task

The **[how-to guides](../how-to/run-the-offline-demo.md)** are one page per task, written for someone who has done the tutorials and does not need them repeated:

- [Run the demo with the wifi switched off](../how-to/run-the-offline-demo.md) — for showing it to someone else
- [Check what a machine can do](../how-to/check-what-this-machine-can-do.md) — before committing an afternoon to it
- [Choose the promise level (α)](../how-to/choose-alpha.md) — and how many photographs it needs
- [Save, load and migrate a learner](../how-to/save-load-and-migrate.md)
- [Fine-tune from corrections](../how-to/fine-tune-with-corrections.md) — the optional deeper adjustment, with the PyTorch extra
- [Use a different backbone](../how-to/use-your-own-backbone.md), [export one to ONNX](../how-to/export-a-backbone-to-onnx.md)
- [Run the benchmarks](../how-to/run-the-benchmarks.md) — reproduce every published number
- [Use Tambua](../how-to/use-tambua.md), the web application built on AdaptShot
- [Deploy offline](../how-to/deploy-offline.md), [troubleshoot](../how-to/troubleshoot.md)

## I want to understand why

The **[Understand](../understand/how-it-works.md)** section explains the design and its trade-offs:

- [How it works](../understand/how-it-works.md) — the pipeline from photograph to answer, in one page
- [The guarantee](../understand/the-guarantee.md) — exactly what the prediction set promises, when it holds, and where it stops holding, with the measurements
- [Why CPU-only and offline](../understand/why-cpu-only-and-offline.md)
- [Algorithm theory](../understand/algorithm-theory.md), for the mathematics
- [The technical note](../understand/technical-note.md) — two pages, every number traceable

## I need the exact details

The **[Reference](../reference/api.md)**: every public name, classified as stable or experimental; every configuration field; every error and what it means; what each results file contains.

## I want to contribute

Start with **[Contributing](../contributing/contributing.md)** and the [development setup](../contributing/development-setup.md). The five-stage validation gate, the API stability policy, how these documentation pages are tested, and how a release happens are each their own page. Every code block in the tutorials and how-tos is executed by the test suite on every change; if you find one that does not run, that is a bug worth reporting.
