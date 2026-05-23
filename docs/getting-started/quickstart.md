# Quick Start

This quick start uses generated images, so it does not need a downloaded dataset. It exercises the unreleased v0.1.1 native workflow in `src/adaptshot/core/learner.py`: load support images, predict, route a correction, and profile latency plus memory.

!!! note "String Labels Are Supported"
    In the v0.1.1 branch, `FewShotLearner.correct()` accepts string or integer labels. You can keep human-readable labels in the UI and map them directly into the learner.

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

    config = AdaptShotConfig(
        backbone="resnet18",
        device="cpu",
        seed=42,
        max_buffer_size=10,
        use_faiss=False,
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
    print(f"Latency: {latency_ms:.1f} ms")
    print(f"Peak traced memory: {peak_bytes / 1024 / 1024:.1f} MiB")

    # Simulate a human correction. This intentionally uses integer true_label.
    feedback = learner.correct(
        image_path=str(query_path),
        true_label=0,
        confidence_weight=0.95,
    )
    print(f"Correction routed: {feedback['calibration_updated']}")
    print(f"Fine-tuned: {feedback['fine_tuned']}")
```

Example output will resemble:

```text
Prediction: maize_blight
Calibrated confidence: 99.7%
Needs review: False
Latency: 150.6 ms
Peak traced memory: 0.5 MiB
Correction routed: True
Fine-tuned: False
```

!!! note "About The Numbers"
    The tutorial measures latency and traced Python allocations on your machine. Do not treat the example output as a benchmark claim. For the supported benchmark harness, see [Benchmarks](benchmarks.md).

## Step 3: Save State

```python
learner.save("checkpoints/demo.json")
```

This creates:

- `checkpoints/demo.json`
- `checkpoints/demo.embeddings.npy`
- `checkpoints/demo.head.pt`

!!! warning "Load Caveat"
    The v0.1.1 branch includes `FewShotLearner.load(path)` with checkpoint integrity validation and schema migration support.

## Verification Checklist

- [ ] The script imports `FewShotLearner` and `AdaptShotConfig`.
- [ ] `load_support_images(image_paths, labels)` receives lists with matching length.
- [ ] `predict()` prints a `PredictionResult`.
- [ ] `correct()` uses integer labels and returns a dictionary with `fine_tuned`.
- [ ] Latency and memory are measured locally with `time.perf_counter()` and `tracemalloc`.
