---
title: "13 — MziziGuard: Crop Disease Detection Demo"
nav_order: 13
---

# MziziGuard: Crop Disease Detection with AdaptShot

This tutorial walks through a complete, self-contained demo that showcases AdaptShot's core value: helping smallholder farmers identify crop diseases from just a few photos — no internet, no GPU, no expensive hardware.

**You will learn:**
- How few-shot learning works in practice (5 photos per disease class)
- How to use human-in-the-loop corrections with `FewShotLearner.correct()`
- How OOD detection prevents wrong answers in the field
- How to read calibration reports for system monitoring
- How to present AdaptShot to non-technical audiences

---

## The Problem

In Tanzania and across East Africa, maize is the staple food. But diseases like **Northern Leaf Blight** and **Gray Leaf Spot** destroy 20–60% of harvests every season. Agricultural extension officers can't reach every village. By the time a farmer gets a diagnosis, the crop is already lost.

Almost every farmer has a basic smartphone. What if they could just take a photo and get an instant, accurate diagnosis — without needing internet?

That's what MziziGuard demonstrates.

---

## Running the Demo

The demo is fully self-contained — it generates synthetic leaf images so there are **zero external dependencies** beyond AdaptShot itself.

```bash
# Install AdaptShot if you haven't already
pip install adaptshot

# Run the demo (interactive mode with pause prompts)
python examples/crop_disease_demo.py

# Run without pause prompts (for testing)
python examples/crop_disease_demo.py --no-pause
```

The demo has **6 stages**, each telling a piece of the story:

| Stage | What happens | Key concept |
|-------|-------------|-------------|
| 0 | Why this matters | The problem |
| 1 | Load 5 photos per disease | **Few-shot learning** |
| 2 | Predict on a new photo | **Inference** |
| 3 | A human corrects the model | **Human-in-the-loop** |
| 4 | Show something unfamiliar | **OOD detection** |
| 5 | System health report | **Calibration** |
| 6 | Why AdaptShot, why Tanzania | **Vision** |

---

## How It Works (Step by Step)

### 1. Generating the Dataset

The demo uses PIL to create synthetic leaf images — green ovals with veins for healthy leaves, plus brown lesions (Northern Leaf Blight) or gray rectangular spots (Gray Leaf Spot). This means you can run the demo anywhere, even without downloading real datasets.

```python
from examples.crop_disease_demo import generate_dataset, DISEASE_GENERATORS

# Generate 5 support + 3 query images per class
paths, labels = generate_dataset("/tmp/demo", n_support=5, n_query=3)
# 3 classes × 5 support = 15 training images total
```

### 2. Creating the Learner

```python
from adaptshot import AdaptShotConfig, FewShotLearner

config = AdaptShotConfig(
    backbone="resnet18",   # Frozen feature extractor
    device="cpu",           # No GPU needed
    seed=42,                # Reproducible
    eco_mode=True,          # Carbon-aware inference
)
learner = FewShotLearner(config=config)
learner.load_support_images(paths, labels)
```

Behind the scenes, AdaptShot extracts embeddings from each support image using a frozen ResNet-18. These embeddings become the model's "knowledge" — it compares new images to these stored examples.

### 3. Making a Prediction

```python
result = learner.predict("path/to/query_leaf.png")

print(result.prediction)           # "northern_leaf_blight"
print(result.calibrated_confidence) # 0.87
print(result.uncertainty_flag)     # False (model is confident)
```

The `PredictionResult` tells you more than just a label. It gives you a **calibrated confidence** score — the model tells you HOW sure it is, not just WHAT it thinks.

### 4. Human Correction (The Magic)

This is what makes AdaptShot different from every other ML library:

```python
# The agricultural officer says: "This is actually northern_leaf_blight"
result = learner.correct(
    image_path="path/to/query_leaf.png",
    true_label="northern_leaf_blight",
    confidence_weight=0.8,  # How confident is the human?
)
```

The correction is routed through AdaptShot's feedback pipeline:
1. Updates the calibration engine (temperature scaling)
2. Adds the example to the replay buffer
3. Updates per-class ACT thresholds
4. Triggers CA-EWC fine-tuning if enough corrections accumulate

**Every correction makes the model smarter for the next farmer.**

### 5. OOD Detection — Saying "I Don't Know"

```python
# Show it something completely unfamiliar (soil, a hand, a rock)
result = learner.predict("path/to/soil_photo.png")

if result.ood_flag:
    print("I don't know what this is.")  # Honest, not guessing
```

Most AI systems would confidently give a wrong answer. AdaptShot's OOD (out-of-distribution) detection checks whether the new image is close enough to any known support example. If not, it raises a flag instead of guessing.

### 6. System Health Monitoring

```python
report = learner.calibration_report()

print(report["ece"])              # Expected Calibration Error
print(report["temperature"])      # Current temperature scaling
print(report["window_size"])      # Number of corrections recorded
print(report["support_size"])     # Total support embeddings
```

You don't need to be an ML expert to know if the system is healthy. The calibration report gives you plain numbers that tell you whether the model is well-calibrated and has enough data.

---

## Adapting MziziGuard for Your Own Use Case

The demo is designed to be a template. Here's how to adapt it:

### Swap the image generator for real data

Replace the synthetic leaf generators with your own PIL images:

```python
from PIL import Image

def load_my_images(class_name, folder):
    paths, labels = [], []
    for fname in os.listdir(folder):
        if fname.endswith((".jpg", ".png")):
            paths.append(os.path.join(folder, fname))
            labels.append(class_name)
    return paths, labels

# Load 5-10 images per class from your folders
all_paths, all_labels = [], []
for class_name in ["healthy_crop", "disease_a", "disease_b"]:
    p, l = load_my_images(class_name, f"data/{class_name}/")
    all_paths.extend(p)
    all_labels.extend(l)
```

### Change the disease information

Update the `DISEASE_INFO` dictionary in the demo script to match your use case:

```python
DISEASE_INFO = {
    "healthy_crop": {
        "swahili": "mazao yenye afya",
        "action": "No treatment needed.",
        "impact": "Your crop is healthy!",
    },
    # Add your own diseases here...
}
```

### Other use cases that work with the same template

- **Coffee leaf rust detection** — major issue for Tanzania's cash crop
- **Cassava mosaic disease** — affects food security across East Africa
- **Poultry disease screening** — respiratory conditions from droppings photos
- **Skin condition triage** — community health workers in rural clinics
- **Manufacturing defect detection** — quality control for small industries

Any problem where a non-expert needs to classify images with just a few examples is a candidate for AdaptShot.

---

## Presentation Tips

If you're presenting this demo to a non-technical audience:

1. **Start with the problem, not the technology.** "How many of you know a farmer who lost crops to disease?" Everyone will raise their hand.

2. **Use Swahili names.** The demo includes Swahili translations for each disease. Say "ugonjwa wa mabaka ya kahawia" alongside "Northern Leaf Blight."

3. **Emphasize the "just 5 photos" part.** Most people think AI needs millions of examples. "You don't need a thousand pictures of sick maize. You need five."

4. **The human correction moment is your climax.** Pause. Explain: "In most AI, if the computer is wrong, you're stuck. In AdaptShot, you correct it — and it learns. Like teaching a student."

5. **The OOD demo is your trust-builder.** "What if someone shows it a photo of their hand by accident? Most AI would say 'diseased crop.' AdaptShot says 'I don't know.' That's the difference between a tool you can trust and a tool you can't."

6. **End with the big picture.** "One laptop. One extension officer. Every farmer in the district gets a crop doctor. That's what we're building."

---

## What This Demo Teaches About AdaptShot

| Feature | How the demo shows it |
|---------|----------------------|
| Few-shot learning | Only 5 images per disease class |
| CPU-only | Runs on the presenter's laptop, no GPU |
| Offline capable | No internet connection needed |
| Human-in-the-loop | Officer corrects a wrong prediction |
| Calibrated confidence | Shows confidence %, not just labels |
| OOD detection | Refuses to classify soil/non-leaf images |
| Eco mode | `eco_mode=True` in config |
| Calibration report | `calibration_report()` at the end |
| Zero external deps | Synthetic images via PIL, no dataset download |

---

## Next Steps

- Try the [Human-in-the-Loop Deep Dive](../guides/human-in-the-loop.md) for a deeper technical walkthrough
- Read [Real-World Use Cases](../guides/real-world-use-cases.md) for more application ideas
- Explore the [Beginner 101](../getting-started/beginner-101.md) guide if you're new to few-shot learning
- Check the [Full API Reference](../api/core.md) for all `FewShotLearner` methods
