# Use a different backbone

> **For:** someone who suspects the bundled encoder is the limit on their task and wants to try another. Assumes the tutorials. Needs the PyTorch extra for anything beyond the bundled backbone.

<!-- needs: torch -->

## What a backbone is here

The *backbone* is the frozen network that turns a photograph into a list of numbers. AdaptShot never trains it; everything it learns sits on top. The standard install bundles one, `mobilenet_v3_small` (4 MB, ImageNet-pretrained), exported to ONNX so no PyTorch is needed to run it.

Two backbones are registered:

| name | size | needs | notes |
|---|---|---|---|
| `mobilenet_v3_small` | 4.0 MB | nothing — bundled | the default; the one every published number was measured with in 0.3.0 |
| `resnet18` | 44.8 MB | the torch extra | not bundled because of its size; torchvision fetches its weights on first use |

## Switch

```python
from adaptshot import AdaptShotConfig, FewShotLearner
from adaptshot.data import sample_images

paths, labels = sample_images()
learner = FewShotLearner(config=AdaptShotConfig(backbone="resnet18"))  # docs: not run -- downloads resnet18 weights on first use
learner.load_support_images(paths[:-1], labels[:-1])
print(learner.predict(paths[-1]).prediction)
```

The first use of `resnet18` downloads its weights through torchvision (once, cached under `~/.cache/torch`), which means it needs the network that first time. Everything after is offline.

On a standard install without torch, the same code raises `BackboneError` naming both ways out: use a bundled backbone, or install `adaptshot[torch]`. It does not raise `ImportError: torch` from four frames inside the library.

## Should you expect it to help?

On the CIFAR-10 smoke split (25 queries) the two scored 68% and 76% — intervals that overlap heavily, so at that size they are indistinguishable — and the larger backbone was slower. The PlantVillage result was measured on the bundled backbone only. Your task may differ. Measure it: [run the benchmarks](run-the-benchmarks.md) with `--backbone`, or hold out photographs as in [tutorial 3](../tutorials/03-your-own-photos.md) and compare.

## Saved learners and backbones

A saved learner records the backbone its numbers came from and can only be loaded meaningfully on an install that provides it. A `resnet18` learner needs torch wherever it is loaded; a `mobilenet_v3_small` learner loads anywhere. See [save, load and migrate](save-load-and-migrate.md).

## Adding a backbone that is not registered

`BackboneRegistry` in `adaptshot.core.extractor` maps a name to a torchvision constructor, and `BACKBONE_OUTPUT_DIM` records its embedding width. Adding an entry there, and to the `Backbone` type in `adaptshot.config.settings`, registers a new torch backbone. To run it without torch, [export it to ONNX](export-a-backbone-to-onnx.md). Both are contributions rather than configuration; the [contributing guide](../contributing/contributing.md) covers the tests that go with them.
