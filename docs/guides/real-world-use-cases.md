# Real-World Use Cases

AdaptShot was built for resource-constrained environments. This guide shows concrete, runnable examples of how the library solves real problems in agriculture, healthcare, conservation, and manufacturing.

Every example in this guide uses the same core API (`FewShotLearner`) and runs CPU-only with <250MB RAM. No GPU. No cloud. No large datasets.

---

## 1. Agriculture: Crop Disease Detection

**Problem**: A small-scale farmer needs to identify cassava mosaic virus from leaf photos. They have 10 reference images, a $50 Android phone, and no internet in the field.

**Solution**: Classify a field photo against a small support set of healthy and diseased leaves. Route uncertain predictions to a human expert for verification.

### Runnable Example

```python
"""Crop disease detection -- 10 support images, 2 classes."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig


def make_leaf_image(path: Path, green: int, brown: int, seed: int) -> None:
    """Create a synthetic leaf-like image (green=healthy, brown=blighted)."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 0] = brown       # Red channel (brown tint)
    arr[:, :, 1] = green       # Green channel
    arr[:, :, 2] = rng.integers(0, 40, size=(224, 224), dtype=np.uint8)
    noise = rng.integers(0, 25, size=(224, 224, 3), dtype=np.uint8)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


with tempfile.TemporaryDirectory(prefix="crop_disease_") as tmp:
    root = Path(tmp)
    (root / "healthy").mkdir()
    (root / "blight").mkdir()

    image_paths: list[str] = []
    labels: list[str] = []

    # 5 healthy leaves
    for i in range(5):
        p = root / "healthy" / f"leaf_{i}.png"
        make_leaf_image(p, green=180, brown=20, seed=i)
        image_paths.append(str(p))
        labels.append("healthy")

    # 5 blighted leaves
    for i in range(5):
        p = root / "blight" / f"leaf_{i}.png"
        make_leaf_image(p, green=60, brown=160, seed=100 + i)
        image_paths.append(str(p))
        labels.append("blighted")

    # Field photo (blighted)
    field = root / "field_photo.png"
    make_leaf_image(field, green=55, brown=170, seed=999)

    # Configure for field deployment
    config = AdaptShotConfig(
        backbone="mobilenet_v3_small",  # Lightweight for phone CPU
        device="cpu",
        seed=42,
        max_buffer_size=50,
        eco_mode=True,                   # Save battery
        early_exit_threshold=0.90,
    )
    learner = FewShotLearner(config=config)
    learner.load_support_images(image_paths, labels)

    # Diagnose
    result = learner.predict(str(field))
    print(f"Crop diagnosis: {result.prediction}")
    print(f"Confidence: {result.calibrated_confidence:.1%}")

    if result.uncertainty_flag:
        # Route to agricultural extension officer for review
        print("Uncertain diagnosis -- routing to human expert.")
        learner.correct(
            image_path=str(field),
            true_label="blighted",
            confidence_weight=0.95,  # Expert is very sure
        )
        print("Correction recorded. Model will improve for next diagnosis.")
```

**Key Takeaways**:
- `mobilenet_v3_small` keeps the model small enough for phone deployment
- `eco_mode=True` extends battery life in the field
- Human corrections accumulate over time, improving accuracy for the entire community

---

## 2. Healthcare: Pneumonia Triage from Chest X-Rays

**Problem**: A rural clinic has a basic X-ray machine but no radiologist. The nurse needs to flag urgent pneumonia cases for transfer to a city hospital versus routine cases that can wait.

**Solution**: Train AdaptShot on 15 labeled X-rays (10 pneumonia, 5 normal) and use it to triage new cases. High-confidence pneumonia predictions get flagged for immediate transfer.

### Usage Pattern

```python
# Support set: 10 pneumonia X-rays, 5 normal X-rays
xray_paths = [
    "xrays/pneumonia_01.png", "xrays/pneumonia_02.png",  # ... 10 total
    "xrays/normal_01.png", "xrays/normal_02.png",        # ... 5 total
]
xray_labels = (["pneumonia"] * 10) + (["normal"] * 5)

config = AdaptShotConfig(
    backbone="resnet18",           # Higher accuracy for medical use
    calibration_method="scaling_binning",  # Finer calibration
    enable_ood_detection=True,     # Flag unusual images
    ood_threshold_quantile=0.95,   # Conservative threshold
    recalibrate_after_feedback=True,
)
learner = FewShotLearner(config=config)
learner.load_support_images(xray_paths, xray_labels)

# Triage a new patient
result = learner.predict("xrays/new_patient.png")

if result.prediction == "pneumonia" and result.calibrated_confidence > 0.85:
    print("URGENT: Transfer to city hospital immediately.")
elif result.uncertainty_flag or result.ood_flag:
    print("Uncertain: Send image to remote radiologist for review.")
else:
    print("Normal: Patient can wait for routine follow-up.")
```

**Key Takeaways**:
- Medical use cases demand high calibration quality -- `scaling_binning` provides finer control than `temperature`
- OOD detection catches images that don't look like any training example (different equipment, patient positioning)
- The human-in-the-loop loop builds a growing corpus of labeled examples

---

## 3. Conservation: Wildlife Camera Trap Classification

**Problem**: A conservation NGO has camera traps in a remote national park. They collect thousands of images but can only label a few dozen manually. They need to classify remaining images as containing endangered species or common animals.

**Solution**: Use a small labeled support set to seed AdaptShot, then run batch prediction on the remaining images. Flag uncertain classifications for human review by a conservation biologist.

### Batch Processing Pattern

```python
from pathlib import Path

# 20 labeled support images (10 endangered, 10 common)
support_paths = list(Path("labeled/endangered").glob("*.jpg"))[:10]
support_paths += list(Path("labeled/common").glob("*.jpg"))[:10]
support_labels = (["endangered"] * 10) + (["common"] * 10)

config = AdaptShotConfig(
    backbone="resnet18",
    device="cpu",
    max_buffer_size=200,        # Larger buffer for batch processing
    use_faiss=True,             # FAISS for faster search with many images
    enable_ood_detection=True,
)
learner = FewShotLearner(config=config)
learner.load_support_images(
    image_paths=[str(p) for p in support_paths],
    labels=support_labels,
)

# Batch predict on unlabeled camera trap images
unlabeled = list(Path("unlabeled").glob("*.jpg"))
results = {
    "endangered_high_conf": [],
    "endangered_needs_review": [],
    "common": [],
    "ood_rejected": [],
}

for img_path in unlabeled:
    result = learner.predict(str(img_path))

    if result.ood_flag:
        results["ood_rejected"].append(str(img_path))
    elif result.prediction == "endangered":
        if result.uncertainty_flag:
            results["endangered_needs_review"].append(str(img_path))
        else:
            results["endangered_high_conf"].append(str(img_path))
    else:
        results["common"].append(str(img_path))

print(f"Endangered (high confidence): {len(results['endangered_high_conf'])}")
print(f"Endangered (needs review):   {len(results['endangered_needs_review'])}")
print(f"Common animals:              {len(results['common'])}")
print(f"OOD / rejected:              {len(results['ood_rejected'])}")
```

**Key Takeaways**:
- FAISS acceleration (`use_faiss=True`) improves batch throughput for large image sets
- Larger `max_buffer_size` accommodates growing support sets as corrections accumulate
- OOD detection prevents false positives on animals never seen before

---

## 4. Manufacturing: Defect Detection from Few Reference Images

**Problem**: A small factory needs to detect manufacturing defects on circuit boards. They have 8 photos of "good" boards and 6 photos of "defective" boards. Defects are rare and varied.

**Solution**: Train AdaptShot on the small support set and use it to screen new boards. Route defect predictions to a quality control engineer for confirmation.

### Usage Pattern

```python
config = AdaptShotConfig(
    backbone="resnet18",
    inference_mode="prototypical",    # Better with imbalanced classes
    similarity_metric="cosine",       # Direction matters more than magnitude
    calibration_method="temperature",
    recalibrate_after_feedback=True,
)
learner = FewShotLearner(config=config)
learner.load_support_images(
    image_paths=["good_01.jpg", "good_02.jpg", ..., "defect_06.jpg"],
    labels=["good", "good", ..., "defective"],
)

# Screen a new board
result = learner.predict("board_new.jpg")

if result.prediction == "defective":
    print("Potential defect detected -- route to QC engineer.")
    if result.calibrated_confidence < 0.80:
        print("Low confidence -- request immediate human verification.")
elif result.uncertainty_flag:
    print("Uncertain -- flag for manual inspection.")
else:
    print("Pass: board appears defect-free.")
```

**Key Takeaways**:
- `prototypical` inference mode handles imbalanced classes better than nearest-neighbor
- `cosine` similarity focuses on direction (shape features) rather than magnitude (brightness)
- Corrections from QC engineers continuously refine the model

---

## 5. Education: Learning Disability Screening from Handwriting

**Problem**: A rural teacher needs to screen for early signs of dysgraphia (writing difficulty) using samples of student handwriting. They have 12 reference samples and no specialist available on-site.

**Solution**: AdaptShot can classify handwriting samples against known patterns, flagging students who may need specialist evaluation.

### Usage Pattern

```python
config = AdaptShotConfig(
    backbone="mobilenet_v3_small",
    device="cpu",
    max_buffer_size=30,
    early_exit_threshold=0.92,
)
learner = FewShotLearner(config=config)
learner.load_support_images(
    image_paths=[...],  # Handwriting samples as images
    labels=[...],       # "typical", "needs_evaluation"
)

result = learner.predict("student_sample.jpg")
if result.prediction == "needs_evaluation":
    print("Recommend specialist evaluation.")
else:
    print("Writing appears within typical range.")
```

---

## 6. MziziGuard: A Worked Example (Crop Disease)

!!! warning "This is a demonstration, not a deployment"
    MziziGuard has **never been deployed**. No farmer has used it. An earlier version
    of this page described it as deployed and production-ready; that was wrong, and it
    is corrected here.

    Its sample images are **generated procedurally** — see
    `apps/tambua/src/tambua/data.py`, which draws them with `ImageDraw.ellipse()`. Any
    accuracy you observe on them measures the classifier's ability to tell drawn shapes
    apart, and nothing else.

    The application is being rebuilt as
    [Tambua](https://github.com/johnson2006christopher/adaptshot/issues/45), a general
    few-shot application, and evaluated against real photographs
    ([#18](https://github.com/johnson2006christopher/adaptshot/issues/18)). This section
    will be replaced when that lands.

MziziGuard is a complete Gradio application with 5 interactive tabs, built entirely on AdaptShot. It runs on a standard laptop with no internet, and demonstrates the full API end to end: support-set loading, prediction, OOD rejection, and human correction.

### Architecture

```
Farmer takes photo → MziziGuard Web UI → MziziGuard Engine → AdaptShot FewShotLearner
                                                                    → ResNet-18 Backbone
                                                                    → Similarity Search
                                                                    → Calibration + ACT + OOD
                                                                    → Identification (Swahili + Action)
```

### Key Numbers

| Property | Value | Backed by |
|----------|-------|-----------|
| Training images needed | 5 per disease class | `apps/tambua/src/tambua/configs/maize.yaml` |
| Supported crops | Configurable via YAML | `apps/tambua/src/tambua/configs/maize.yaml` |
| Languages | English + Swahili | `localization` block in the config |
| Internet required | No (after install) | no network calls after backbone download |

!!! note "Removed from this table"
    It previously listed `Memory usage: <250MB` and `Inference latency: ~150ms on CPU`.
    Neither figure was produced by any script in `benchmarks/`. They have been removed
    rather than re-estimated. Memory is tracked in
    [#13](https://github.com/johnson2006christopher/adaptshot/issues/13) and latency in
    [#20](https://github.com/johnson2006christopher/adaptshot/issues/20); both will
    return with hardware and seed recorded alongside them.

### How to Run It

```bash
pip install tambua
tambua
# → http://localhost:7860
```

Full guide: [Tambua Complete Guide](tambua-complete-guide.md)

---

## General Patterns Across All Use Cases

| Concern | Config Choice | Why |
|---------|--------------|-----|
| Battery life | `eco_mode=True`, `mobilenet_v3_small` | Reduces compute and energy |
| Medical accuracy | `resnet18`, `scaling_binning` | Higher accuracy, finer calibration |
| Many images | `use_faiss=True` | FAISS accelerates batch search |
| Rare classes | `inference_mode="prototypical"` | Better with imbalanced support sets |
| Unknown inputs | `enable_ood_detection=True` | Flags images outside known distribution |
| Growing knowledge | `recalibrate_after_feedback=True` | Model improves with every correction |

---

## Profiling and Monitoring Across Use Cases — v0.2.0

Every deployment should incorporate production monitoring. Here is how to add profiling to any use case:

### Add Memory Tracking

```python
from adaptshot.profiling import MemoryTracker

tracker = MemoryTracker()
tracker.start()

# In your inference loop:
with tracker.section("batch_inference"):
    for img_path in unlabeled_images:
        result = learner.predict(img_path)

report = tracker.get_report()
if report['peak_memory_mb'] > 250:
    print(f"⚠️ Memory budget exceeded: {report['peak_memory_mb']:.0f} MB")
```

### Add Penalty Trend Monitoring

```python
# After each correction, check penalty trends
if hasattr(learner, 'explainability_engine'):
    penalties = learner.explainability_engine.get_penalty_summary()
    if penalties.get('global_trend') == 'degrading':
        print("⚠️ Model penalties are increasing — consider refreshing support set")
```

### Periodic Cache Clearing

```python
# For long-running services (conservation camera traps, manufacturing QC):
# Clear backbone cache every 1000 predictions
if prediction_count % 1000 == 0:
    learner.clear_backbone_cache()
```

### Use Case Monitoring Matrix

| Use Case | Key Metric | Alert Threshold | Action |
|----------|-----------|----------------|--------|
| Crop Disease | Memory peak | > 200 MB | Reduce `max_buffer_size` |
| Medical Triage | ECE | > 0.10 | Recalibrate with `scaling_binning` |
| Camera Traps | Penalty trend | "degrading" | Refresh support set |
| Manufacturing | OOD rate | > 20% | Add more "defective" examples |
| Education | Latency | > 500 ms | Switch to `mobilenet_v3_small` |

---

## Verification Checklist

- [ ] You can run the crop disease example on your machine.
- [ ] You understand how to adapt the batch processing pattern to your own dataset.
- [ ] You know which config changes to make for each domain (medical vs. field vs. factory).
- [ ] You understand that AdaptShot's role is to augment human expertise, not replace it.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
