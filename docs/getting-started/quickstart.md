# Quick Start

This quick start uses generated images, so it does not need a downloaded dataset. It exercises the v0.2.0 workflow: load support images, predict with conformal sets, inspect uncertainty, and route corrections.

!!! note "v0.2.0 Conformal & Uncertainty"
    In v0.2.0, `predict()` returns conformal prediction sets, multi-signal uncertainty reports, and explainability results. Calibration uses true leave-one-out conformal and bootstrap temperature estimation for cold starts.

## Step 1: Install

```bash
pip install adaptshot
```

## Step 2: Run A Complete Synthetic Example

Save this as `quickstart_adaptshot.py` and run it with `python quickstart_adaptshot.py`.

```python
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
from PIL import Image

from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig


LABEL_NAMES = {
    "0": "maize_healthy",
    "1": "maize_blight",
}


def make_image(path: Path, base_color: tuple[int, int, int], noise_seed: int) -> None:
    rng = np.random.default_rng(noise_seed)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :] = np.array(base_color, dtype=np.uint8)
    noise = rng.integers(0, 30, size=(224, 224, 3), dtype=np.uint8)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


with tempfile.TemporaryDirectory(prefix="adaptshot_quickstart_") as tmp:
    root = Path(tmp)
    healthy_dir = root / "healthy"
    blight_dir = root / "blight"
    healthy_dir.mkdir()
    blight_dir.mkdir()

    image_paths: list[str] = []
    labels: list[int] = []

    for i in range(3):
        path = healthy_dir / f"healthy_{i}.png"
        make_image(path, (40, 150, 50), noise_seed=i)
        image_paths.append(str(path))
        labels.append(0)

    for i in range(3):
        path = blight_dir / f"blight_{i}.png"
        make_image(path, (150, 80, 35), noise_seed=100 + i)
        image_paths.append(str(path))
        labels.append(1)

    query_path = root / "field_photo.png"
    make_image(query_path, (150, 80, 35), noise_seed=999)

    # v0.2.0 config with all production features
    config = AdaptShotConfig(
        backbone="resnet18",
        device="cpu",
        seed=42,
        max_buffer_size=10,
        use_faiss=False,
        conformal_alpha=0.10,          # 90% coverage guarantee
        explainability_enabled=True,
    )
    learner = FewShotLearner(config=config)
    learner.load_support_images(image_paths=image_paths, labels=labels)

    tracemalloc.start()
    start = time.perf_counter()
    result = learner.predict(str(query_path))
    latency_ms = (time.perf_counter() - start) * 1000
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    prediction_name = LABEL_NAMES[str(result.prediction)]
    print(f"Prediction: {prediction_name}")
    print(f"Calibrated confidence: {result.calibrated_confidence:.1%}")
    print(f"Needs review: {result.uncertainty_flag}")
    print(f"Conformal set: {result.conformal_set}")
    print(f"Latency: {latency_ms:.1f} ms")
    print(f"Peak traced memory: {peak_bytes / 1024 / 1024:.1f} MiB")

    # v0.2.0: Uncertainty report
    if result.uncertainty_report:
        print(f"Uncertainty: epistemic={result.uncertainty_report.get('epistemic', 0):.3f}, "
              f"aleatoric={result.uncertainty_report.get('aleatoric', 0):.3f}")

    # Simulate a human correction
    feedback = learner.correct(
        image_path=str(query_path),
        true_label=0,
        confidence_weight=0.95,
    )
    print(f"Correction routed: {feedback['calibration_updated']}")
    print(f"Fine-tuned: {feedback['fine_tuned']}")
```

Example output:

```text
Prediction: maize_blight
Calibrated confidence: 97.2%
Needs review: False
Conformal set: [1, 0]
Latency: 150.6 ms
Peak traced memory: 0.5 MiB
Uncertainty: epistemic=0.042, aleatoric=0.118
Correction routed: True
Fine-tuned: False
```

!!! note "About The Numbers"
    This tutorial measures latency and traced Python allocations on your machine. Do not treat the example output as a benchmark claim. For the supported benchmark harness, see [Benchmarks](benchmarks.md).

## Step 3: Save State

```python
learner.save("checkpoints/demo.json")
```

This creates:

- `checkpoints/demo.json`
- `checkpoints/demo.embeddings.npy`
- `checkpoints/demo.head.pt`

!!! note "v0.2.0 Checkpoint Integrity"
    `load()` performs SHA-256 integrity verification, schema version migration, and atomic writes. Checkpoints created in v0.1.x are automatically migrated to v0.2.0 format.

## Step 4: Try Contrastive Inference

v0.2.0 supports gradient-trained contrastive prototypes. Switch inference mode:

```python
config = AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    inference_mode="contrastive",  # Use InfoNCE-trained prototypes
    contrastive_config=ContrastiveConfig(
        projection_dim=128,
        temperature=0.1,
        learning_rate=0.01,
    ),
)
learner = FewShotLearner(config=config)
# ... rest is identical to the prototypical flow
```

## Verification Checklist

- [ ] The script imports `FewShotLearner` and `AdaptShotConfig`.
- [ ] `load_support_images(image_paths, labels)` receives lists with matching length.
- [ ] `predict()` returns a `PredictionResult` with `conformal_set` and `uncertainty_report`.
- [ ] `correct()` accepts integer labels and returns a dictionary with `fine_tuned`.
- [ ] Latency and memory are measured locally with `time.perf_counter()` and `tracemalloc`.
