# Save, load and migrate a learner

> **For:** someone who has a taught learner and wants it back tomorrow, on another machine, or after upgrading AdaptShot. Five minutes.

## Save

```python
from adaptshot import FewShotLearner
from adaptshot.data import sample_images

paths, labels = sample_images()
learner = FewShotLearner()
learner.load_support_images(paths, labels)
learner.save("maize.json")
```

Two files appear: `maize.json` (labels, calibration state, thresholds, a schema version and a SHA-256 checksum of the embeddings) and `maize.embeddings.npy` (the numbers computed from every teaching photograph and every correction). Together they are typically well under a megabyte for a few dozen photographs. **Keep them together**; one is useless without the other.

The photographs themselves are not saved. Only what was computed from them.

## Load

```python
from adaptshot import FewShotLearner

restored = FewShotLearner.load("maize.json")
print(restored.predict(paths[0]).prediction)
```

The restored learner is the saved one: same prototypes, same calibration, same corrections. It runs with whatever backbone the current install provides for the name recorded in the file — the standard install has `mobilenet_v3_small` bundled.

## Move it to another machine

Copy both files. Install AdaptShot there. Load. The file records which backbone the embeddings came from; if that backbone is not available on the new machine (for instance `resnet18` without the torch extra), the first `predict()` raises `BackboneError` naming what would work — the load itself succeeds, since it reads numbers rather than photographs. [Check the machine first](check-what-this-machine-can-do.md).

## Migrate after an upgrade

Files saved by an older AdaptShot load with a `RuntimeWarning` saying they were migrated from their schema version to the current one:

```python
import warnings
from adaptshot import FewShotLearner

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    restored = FewShotLearner.load("maize.json")
print([str(w.message) for w in caught])   # empty for a file saved by this version
```

Save again to write the current format. The migration never drops data; it fills in fields that did not exist in the older version with their defaults.

## When loading fails

| error | meaning | what to do |
|---|---|---|
| `AdaptShotError: ... corrupted` | the embeddings file does not match the checksum in the JSON | restore both files from a backup; do not load a mismatched pair |
| `AdaptShotError: Failed to read embeddings file` | the two files were separated, or the `.npy` was damaged | put them back in the same folder, or restore from a backup |
| `BackboneError` on the first `predict()` | the backbone the file was made with is not available here | install `adaptshot[torch]`, or re-teach on this machine |

A learner that fails to load never half-loads: you get the error and no object, so there is nothing partially initialised to mislead you.

## What is not in the file

The learner's *configuration* is saved, so the same α, thresholds and calibration apply. The optional fine-tuned head from the [torch extra](fine-tune-with-corrections.md) is **not** saved: it is rebuilt fresh on load. The corrections that trained it *are* saved, so on an install with torch it is retrained from them as corrections continue.
