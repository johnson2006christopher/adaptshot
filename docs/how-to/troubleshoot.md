# Troubleshoot

> **For:** someone who has hit an error or a result that looks wrong. Find your symptom in the left column. Every error AdaptShot raises on purpose derives from `AdaptShotError`, so `except AdaptShotError` catches the library and nothing else.

## Errors

| you see | it means | do this |
|---|---|---|
| `ConfigValidationError: image_paths and labels must have the same length` | the two lists you passed to `load_support_images` do not line up | build both lists in the same loop, as in [tutorial 3](../tutorials/03-your-own-photos.md) |
| `ConfigValidationError: conformal_alpha must be in (0.0, 1.0)` | α outside its range | 0.10 for a 90% promise; see [choose α](choose-alpha.md) |
| `InvalidImageError` | a path does not exist, the file is not an image, or it is not readable as RGB | check the path; open the file in an image viewer; convert exotic formats to JPEG or PNG |
| `BackboneError: Backbone 'resnet18' needs PyTorch on this install` | you asked for a backbone whose weights are not bundled | use `mobilenet_v3_small`, or `pip install "adaptshot[torch]"` — the message names both |
| `AdaptShotError: ... corrupted` on `load()` | the saved JSON and its `.embeddings.npy` do not match | restore both from a backup; see [save, load and migrate](save-load-and-migrate.md) |
| `AdaptShotError: Failed to read embeddings file` | the `.embeddings.npy` is missing or damaged | keep the two files together |
| `BufferCapacityError` | pruning the correction buffer failed; a deterministic fallback was applied and the error is reported | file an issue with the traceback — this should not happen |
| `ModuleNotFoundError: No module named 'adaptshot'` | the virtual environment is not active, or the install did not finish | activate it ([tutorial 1](../tutorials/01-install.md), step 3), then `pip install adaptshot` |
| `ModuleNotFoundError: No module named 'torch'` from *your* code | you imported torch yourself on a standard install | the library never needs it for inference; remove the import or install the extra |

## Warnings

| you see | it means | do this |
|---|---|---|
| `ConformalEngine(alpha=0.050): prediction sets are uninformative ... until 19 calibration scores exist` | you asked for a promise the number of photographs cannot support | more photographs, or a larger α; [choose α](choose-alpha.md) |
| `OOD detection disabled: no class has enough samples` | some class has fewer than three teaching photographs | at least three per class; five is the sensible floor |
| `Checkpoint schema 0.1.0 loaded; migrating` | an older saved learner was upgraded on load | save it again to write the current format |
| `CUDA requested but not available` | `device="cuda"` on a machine without a usable GPU | leave `device` at its default; the defaults are CPU on purpose |

## Results that look wrong

**Every prediction is the same class.** The teaching set is probably unbalanced or mislabelled: one folder with many photographs and others with few, or a folder containing the wrong thing. `result.nearest_neighbors` names the teaching photographs each answer resembled — the fastest way to find a misfiled one.

**Confidence is always low.** Expected at first: calibration learns from corrections, and a fresh learner is deliberately not over-confident. It also means the classes genuinely look alike to the encoder. Check `prototype_margin`: a small margin on every query says the prototypes are close together, and more or clearer photographs help more than any setting.

**The prediction set always has one label.** Either `conformal_calibrated` is `False` (too few photographs for that α — the warning above) or the classes are well separated and one label is the right answer. Check the flag first.

**The prediction set is always every class.** It cannot narrow the photograph down at that α. Treat as a refusal. If it happens on the *teaching* photographs, something is wrong with them.

**`ood_flag` fires on photographs that are clearly in scope.** With very few teaching photographs the threshold is set from little evidence; add photographs. In 0.3.0 the threshold is calibrated leave-one-out and fires on about 1.5% of in-distribution photographs on the benchmark; if you see much more, the teaching photographs may not resemble the queries — different camera, light or background — which is the [distribution shift](../understand/the-guarantee.md) the guarantee does not cover.

**Results differ between two runs.** They should not: every source of randomness is seeded from `config.seed`, and the benchmark checks this on every change. If two runs on the same machine with the same photographs disagree, that is a bug — file it with both outputs.

**It is slow.** The first prediction after building a learner includes loading the model (~1 s). After that, tens of milliseconds per photograph on a laptop; `check_environment()` measures yours. A machine under load reports slower, honestly.

## Before you file an issue

Paste the output of `python -c "import adaptshot; print(adaptshot.check_environment())"`, the full traceback, and — if you can — the smallest script that reproduces it. The [bundled photographs](../tutorials/02-first-prediction.md) make a reproduction possible without sharing your own.
