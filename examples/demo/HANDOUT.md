# AdaptShot — a few-shot image classifier that knows when it doesn't know

**Teach it a new class from a handful of photographs. Get back not just a label,
but a prediction set with a coverage guarantee — and a refusal when it should
refuse.** CPU only. Runs offline. 3.5 MB.

## Try it in the next ten seconds

```
pip install adaptshot
```

```python
from adaptshot import FewShotLearner
from adaptshot.data import sample_images

paths, labels = sample_images()                      # twelve real maize-leaf photos ship in the wheel
learner = FewShotLearner()
learner.load_support_images(paths[:-1], labels[:-1])
result = learner.predict(paths[-1])
print(result.prediction, f"{result.calibrated_confidence:.0%}", result.conformal_set)
```

Measured from a clean virtualenv: install **4.5 s**, first prediction **0.4 s**, no network after `pip`.

## What is measured, not claimed

PlantVillage crop-disease photographs, 5-way 5-shot, 100 episodes, seed 42, CPU:

| | |
|---|---|
| Accuracy | **91.4% ± 1.0** — identical to a nearest-centroid baseline, and we say so |
| Conformal coverage at a 90% target | **97.5% ± 0.7**, mean set size 2.05 |
| The same target, top-1 with a calibrated threshold | 83.9% — it misses the target; conformal clears it |
| Peak memory, full cycle, core install | 120 MB |

The accuracy is not the point. The prediction set and the abstention are. Every
number here has a script behind it in the repository, and CI fails if the
README's figures drift from the benchmark artifact.

## Where it is

- **Repository:** https://github.com/johnson2006christopher/adaptshot
- **Install:** `pip install adaptshot` (Python 3.10+; numpy, Pillow, onnxruntime)
- **DOI:** _pending — the Zenodo archive is tracked as issue #24; the badge will appear on the repository_
- **Licence:** MIT. Sample photographs are PlantVillage's, CC BY-SA 3.0, cited in the package.

Johnson Christopher Hassan — Tanzania. Built for the field, where the network is
the thing you cannot count on.

_Print note: add a QR code of the repository URL here._
