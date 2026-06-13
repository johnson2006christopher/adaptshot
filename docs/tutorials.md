# Tutorial-Style Guides

These tutorials are designed to be copied, pasted, and run. They use only APIs that exist in AdaptShot v0.2.0.

AdaptShot is still native-first here: use the Python API as the source of truth, then layer optional Studio workflows on top only when you need the browser UI.

!!! warning "Correction Labels In v0.1.0"
    `FewShotLearner.correct()` currently works reliably with integer-like labels because `FeedbackRouter` updates calibration with `int(predicted_label)` and `int(corrected_label)`. Use a label map for display names until string-label corrections are fixed.

## Tutorial 01: Getting Started With Synthetic Images

This tutorial simulates a two-class crop disease task: `maize_healthy` and `maize_blight`. It uses synthetic images so the full prediction and correction workflow can run anywhere.

### Prerequisites

- Python 3.9+
- AdaptShot v0.1.0
- CPU execution

### Step 1: Install

```bash
pip install adaptshot
```

### Step 2: Create Support Images, Predict, And Correct

```python
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
from PIL import Image

from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig


label_names = {
    "0": "maize_healthy",
    "1": "maize_blight",
}


def save_leaf_like_image(path: Path, rgb: tuple[int, int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = np.full((224, 224, 3), rgb, dtype=np.uint8)
    texture = rng.integers(0, 45, size=(224, 224, 3), dtype=np.uint8)
    arr = np.clip(arr + texture, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


with tempfile.TemporaryDirectory(prefix="adaptshot_crop_") as tmp:
    root = Path(tmp)
    support_paths: list[str] = []
    support_labels: list[int] = []

    for class_id, class_name, color in [
        (0, "maize_healthy", (35, 155, 55)),
        (1, "maize_blight", (155, 85, 35)),
    ]:
        class_dir = root / class_name
        class_dir.mkdir()
        for i in range(5):
            path = class_dir / f"{i}.png"
            save_leaf_like_image(path, color, seed=class_id * 100 + i)
            support_paths.append(str(path))
            support_labels.append(class_id)

    query_path = root / "field_photo.png"
    save_leaf_like_image(query_path, (155, 85, 35), seed=900)

    learner = FewShotLearner(
        config=AdaptShotConfig(
            backbone="resnet18",
            device="cpu",
            seed=42,
            max_buffer_size=10,
            use_faiss=False,
        )
    )
    learner.load_support_images(support_paths, support_labels)

    tracemalloc.start()
    start = time.perf_counter()
    result = learner.predict(str(query_path))
    latency_ms = (time.perf_counter() - start) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Prediction: {label_names[str(result.prediction)]}")
    print(f"Confidence: {result.calibrated_confidence:.1%}")
    print(f"Review needed: {result.uncertainty_flag}")
    print(f"Latency: {latency_ms:.1f} ms")
    print(f"Peak traced memory: {peak_bytes / 1024 / 1024:.1f} MiB")

    # Simulated agronomist correction. true_label=0 maps to maize_healthy.
    feedback = learner.correct(
        image_path=str(query_path),
        true_label=0,
        confidence_weight=0.95,
    )
    print(f"Correction routed: {feedback['calibration_updated']}")
    print(f"Fine-tuned: {feedback['fine_tuned']}")
```

Expected output shape:

```text
Prediction: maize_blight
Confidence: 99.7%
Review needed: False
Latency: 150.6 ms
Peak traced memory: 0.5 MiB
Correction routed: True
Fine-tuned: False
```

!!! note "Why Fine-tuned May Be False"
    `FewShotLearner` configures `FeedbackRouter` to trigger fine-tuning after enough pending corrections. With `max_buffer_size=10`, the trigger threshold is `5`, so one correction is routed but does not fine-tune yet.

### Verification Checklist

- [ ] The code starts with `pip install adaptshot`.
- [ ] The support set contains matching `support_paths` and `support_labels`.
- [ ] Prediction succeeds before correction.
- [ ] Correction succeeds with integer labels.
- [ ] Latency and memory are measured on the user's machine.

## Tutorial 02: Human-in-the-Loop Crop Monitoring

See [tutorials/02_human_in_the_loop.md](tutorials/02_human_in_the_loop.md) for the correction loop.

## Tutorial 03: Continual Learning And Persistence

See [tutorials/03_continual_learning.md](tutorials/03_continual_learning.md) for correction buffering, calibration updates, and safe save/load.

## Tutorial 04: Production Readiness And Debugging

See [tutorials/04_production_ready.md](tutorials/04_production_ready.md) for error handling, eco mode, energy profiling, and deployment checks.

## Tutorial 05: Reference And FAQ

See [tutorials/05_reference_faq.md](tutorials/05_reference_faq.md) for the glossary, API table, and troubleshooting quick lookup.

## Tutorial 06: Core API Deep Dive

New users often want one more chapter that walks method-by-method through the library. See the next tutorial for a compact but practical API tour.

## Tutorial 07: Source Tour And Internal Pipeline

See [tutorials/07_source_tour.md](tutorials/07_source_tour.md) for a guided reading of the real prediction pipeline, calibration, ACT, corrections, pruning, and persistence.

## Tutorial 08: Configuration, Determinism, And Safe I/O

See [tutorials/08_configuration_determinism_io.md](tutorials/08_configuration_determinism_io.md) for the control panel, reproducibility helpers, and safe file handling utilities.

## Tutorial 09: Benchmarks And Reproducibility

See [tutorials/09_benchmarks_and_reproducibility.md](tutorials/09_benchmarks_and_reproducibility.md) for the smoke benchmark, energy profiling, and reproducible measurement workflow.

## Tutorial 10: Module Map

See [tutorials/10_module_map.md](tutorials/10_module_map.md) for a guided map of the source tree and which file owns which behavior.

## Tutorial 11: UI Pilot Dashboard

See [tutorials/11_ui_pilot_dashboard.md](tutorials/11_ui_pilot_dashboard.md) for the optional Gradio dashboard that wraps the learner in a browser UI.

## Tutorial 12: Studio Guide

See [tutorials/12_studio_guide.md](tutorials/12_studio_guide.md) for the Gradio Studio Dashboard with full configuration, monitoring, and export features.

---

## v0.2.0 Tutorials

### Tutorial 14: Conformal Prediction

See [tutorials/14_conformal_prediction.md](tutorials/14_conformal_prediction.md) for distribution-free prediction sets with guaranteed coverage.

### Tutorial 15: Advanced Uncertainty Quantification

See [tutorials/15_advanced_uncertainty.md](tutorials/15_advanced_uncertainty.md) for multi-signal uncertainty estimation and OOD detection.

### Tutorial 16: Explainability & XAI

See [tutorials/16_explainability.md](tutorials/16_explainability.md) for interpreting predictions with feature attribution, confidence decomposition, and counterfactuals.

### Tutorial 17: Contrastive Prototype Learning

See [tutorials/17_contrastive_learning.md](tutorials/17_contrastive_learning.md) for InfoNCE-based prototype refinement.

### Tutorial 18: End-to-End Production Workflow

See [tutorials/18_end_to_end_workflow.md](tutorials/18_end_to_end_workflow.md) for a complete production pipeline with monitoring and quality control.

## Tutorial: CIFAR-10 Smoke Benchmark

Use the repository benchmark when you want a public dataset workflow. This downloads CIFAR-10 through torchvision and writes a JSON results file.

### Step 1: Install

```bash
pip install adaptshot
```

If you are working from a cloned repository:

```bash
git clone https://github.com/johnson2006christopher/adaptshot.git
cd adaptshot
pip install -e ".[dev]"
```

### Step 2: Run The Benchmark Harness

```bash
python -m benchmarks.run_benchmark --smoke-test --seed 42 --output results/smoke_test.json
```

The script reports:

- few-shot accuracy on a CIFAR-10 subset
- average latency
- p95 latency
- support embedding time
- determinism check result

Example output shape:

```text
Smoke Test Results:
   Accuracy: 64.0%
   Avg Latency: 110.2 ms
   P95 Latency: 140.3 ms
   Embedding Time: 8.532s
   Results saved to results/smoke_test.json
   Determinism check: PASS
```

!!! warning "Use Local Results"
    The exact numbers depend on CPU, PyTorch build, cached model weights, and disk speed. Cite the generated `results/smoke_test.json` from your own run rather than copying example values.

### Step 3: Inspect The Results

```bash
cat results/smoke_test.json
```

### Verification Checklist

- [ ] CIFAR-10 downloads successfully.
- [ ] `results/smoke_test.json` is created.
- [ ] The JSON contains `accuracy`, `latency_avg_ms`, and `latency_p95_ms`.
- [ ] The determinism check prints `PASS`.

## Tutorial: PlantVillage Folder Layout

AdaptShot v0.1.0 does not include a PlantVillage downloader or folder loader. You can still use PlantVillage images by passing file paths and integer labels directly.

> AdaptShot v0.1.1 supports string labels throughout. Use string labels directly, or use a label map for display names.

### Step 1: Install

```bash
pip install adaptshot
```

### Step 2: Prepare A Small Folder

Use any public PlantVillage download source and arrange a tiny support set like this:

```text
plantvillage_small/
  tomato_healthy/
    001.jpg
    002.jpg
  tomato_late_blight/
    001.jpg
    002.jpg
  query/
    field_photo.jpg
```

### Step 3: Load Paths Explicitly

```python
import time
import tracemalloc

from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig

label_names = {
    "0": "tomato_healthy",
    "1": "tomato_late_blight",
}

support_paths = [
    "plantvillage_small/tomato_healthy/001.jpg",
    "plantvillage_small/tomato_healthy/002.jpg",
    "plantvillage_small/tomato_late_blight/001.jpg",
    "plantvillage_small/tomato_late_blight/002.jpg",
]
support_labels = [0, 0, 1, 1]

learner = FewShotLearner(config=AdaptShotConfig(device="cpu", seed=42))
learner.load_support_images(support_paths, support_labels)

tracemalloc.start()
start = time.perf_counter()
result = learner.predict("plantvillage_small/query/field_photo.jpg")
latency_ms = (time.perf_counter() - start) * 1000
_, peak_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Prediction: {label_names[str(result.prediction)]}")
print(f"Confidence: {result.calibrated_confidence:.1%}")
print(f"Latency: {latency_ms:.1f} ms")
print(f"Peak traced memory: {peak_bytes / 1024 / 1024:.1f} MiB")

feedback = learner.correct(
    image_path="plantvillage_small/query/field_photo.jpg",
    true_label=1,
    confidence_weight=0.9,
)
print(f"Correction routed: {feedback['calibration_updated']}")
```

Expected output shape:

```text
Prediction: tomato_late_blight
Confidence: 78.4%
Latency: 125.6 ms
Peak traced memory: 5.4 MiB
Correction routed: True
```

### Verification Checklist

- [ ] All image paths exist locally.
- [ ] `support_labels` uses integers in v0.1.0.
- [ ] `learner.predict()` runs before `learner.correct()`.
- [ ] The measured latency and memory come from your own machine.
