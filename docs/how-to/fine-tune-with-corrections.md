# Fine-tune from corrections (the PyTorch extra)

> **For:** someone whose corrections keep coming and who wants the model itself to adjust, not only its prototypes and calibration. Assumes the tutorials. Needs the optional PyTorch extra and a machine with a few gigabytes free.

<!-- needs: torch -->

## What fine-tuning adds, and what it does not

Everything in the tutorials — prototypes, calibration, thresholds, corrections — works on the standard install and never changes the model that turns photographs into numbers. That model is frozen: a general-purpose image encoder, bundled, 4 MB.

Fine-tuning trains a small *head* on top of it from your corrections, using CA-EWC — a method that learns from new corrections while penalising changes that would undo earlier ones (the "elastic weight" part), weighted by how confident each correction was (the "correction-aware" part). It helps when your classes are separated by details the general encoder does not emphasise. It does not change the encoder itself, and it cannot create a class you never showed it.

**Be honest with yourself about whether you need it.** On the published benchmark the frozen path is a nearest-centroid classifier, and the layers above it do not change which class comes out; fine-tuning is where that could change, at the cost of a large dependency. Try the standard install first.

## Install the extra

```bash
pip install "adaptshot[torch]"
```

This pulls in PyTorch and torchvision. On Linux the default PyPI wheel is the CUDA build, over 2 GB; if you only want the CPU build, install torch first from its CPU index and then AdaptShot's extra:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install "adaptshot[torch]"
```

`check_environment()` afterwards lists *fine-tuning (CA-EWC) via correct()* under *Available now*.

## It happens on its own

There is no `finetune()` call to make. `correct()` queues each correction; when **five** are pending, the head is fine-tuned on them and the queue clears. Nothing else in your code changes:

```python
from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.data import sample_images, demo_images

paths, labels = sample_images()
learner = FewShotLearner(config=AdaptShotConfig(conformal_alpha=0.10))
learner.load_support_images(paths[:-1], labels[:-1])

# Five corrections trip the fine-tune. Here the "corrections" re-teach known photographs,
# which is enough to see the mechanism; in use they would be photographs it got wrong.
truths = dict(zip(demo_images(), ("healthy_maize", "gray_leaf_spot", "gray_leaf_spot", "tomato"), strict=True))
summaries = [learner.correct(image_path=p, true_label=t) for p, t in list(truths.items())[:3]]
summaries += [learner.correct(image_path=paths[i], true_label=labels[i]) for i in (0, 4)]
print([s["fine_tuned"] for s in summaries])
```

The last summary reports `fine_tuned: True`; the four before it `False`. The threshold is the router's `fine_tune_trigger_threshold` (default 5).

## What it costs

Training runs on the CPU and takes a few seconds for a handful of corrections. Memory: importing PyTorch alone costs several hundred megabytes in the process, so a learner with the extra installed does not meet the 250 MB target — `check_environment()` says so on such a process. If memory is the constraint, do not install the extra.

## What is saved

`save()` stores the corrections and everything derived from them, but **not** the fine-tuned head's weights: on `load()` the head is rebuilt fresh and retrained as corrections continue. See [save, load and migrate](save-load-and-migrate.md).

## Turning it off while keeping torch installed

There is no configuration switch for fine-tuning; it triggers whenever torch is importable and five corrections are pending. If you want torch on the machine but no fine-tuning, use a separate environment without the extra for that learner.
