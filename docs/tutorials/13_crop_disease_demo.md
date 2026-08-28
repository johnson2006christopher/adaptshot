---
title: "13 — MziziGuard: Crop Disease Detection App"
nav_order: 13
---

# MziziGuard: Crop Disease Detection with AdaptShot

MziziGuard is a **complete, working application** built on AdaptShot that helps smallholder farmers identify maize diseases from just a few photos — no internet, no GPU, no expensive hardware.

It comes in two modes:

| Mode | Command | Best for |
|------|---------|----------|
| **Terminal demo** | `python examples/crop_disease_demo.py` | Presentations, quick test |
| **Full web app** | `tambua` | Real usage, training, field deployment |

**You will learn:**
- How to run the full Gradio web application with 5-tab interface
- How few-shot learning works in practice (5 photos per class)
- How to use human-in-the-loop corrections (the core power of AdaptShot)
- How OOD detection prevents wrong answers in the field
- How to read calibration reports for system monitoring
- How to adapt MziziGuard for your own crops and use cases

---

## The Problem

In Tanzania and across East Africa, maize is the staple food. But diseases like **Northern Leaf Blight** and **Gray Leaf Spot** destroy 20–60% of harvests every season. Agricultural extension officers can't reach every village. By the time a farmer gets a diagnosis, the crop is already lost.

Almost every farmer has a basic smartphone. What if they could just take a photo and get an instant, accurate diagnosis — without needing internet?

That's what MziziGuard solves.

---

## Running the Full Web Application

Install dependencies and start the Gradio web server:

```bash
# Install AdaptShot with Gradio UI support
pip install -e ".[ui]"

# Launch the web application
tambua

# Or with a custom port and public link
tambua --port 8080 --share
```

Then open **http://localhost:7860** in your browser. You'll see 5 tabs:

| Tab | What it does |
|-----|-------------|
| ⚙️ **Setup** | Generate synthetic samples or load real images from folders |
| 🔍 **Diagnose** | Upload a crop photo → instant Swahili diagnosis with confidence |
| 👩‍🏫 **Teach** | Correct wrong predictions — the model learns immediately |
| 🏥 **Health** | System calibration dashboard & session metrics |
| 📦 **Batch** | Process multiple photos at once, export results |

### Quick Start: The Terminal Demo

For a 6-stage narrated presentation (great for demos):

```bash
python examples/crop_disease_demo.py
# Press Enter between stages for a live presentation
# Use --no-pause for non-interactive mode
```

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

## Using the Python API

MziziGuard has a clean Python API for programmatic use. Here's the full workflow:

### Configuration (config.yaml)

Edit `apps/tambua/configs/maize.yaml` to add your own crops:

```yaml
crops:
  maize:
    swahili: "mahindi"
    diseases:
      healthy_maize:
        swahili: "mahindi yenye afya"
        action: "Hakuna matibabu yanayohitajika."
        severity: "low"
      northern_leaf_blight:
        swahili: "ugonjwa wa mabaka ya kahawia"
        action: "Ondoa majani yaliyoathirika. Tumia dawa ya kuvu."
        severity: "moderate"
```

### Initialize and Predict

```python
from tambua import MziziGuard

# Load from config
guard = MziziGuard("apps/tambua/configs/maize.yaml")

# Option A: Generate synthetic samples for quick start
guard.initialize_with_samples(n_support=5)

# Option B: Load real images from a folder structure
# (folder/{class_name}/*.png)
guard.load_images_from_dir("path/to/crop_photos/", max_per_class=10)

# Diagnose
result = guard.diagnose("photo_of_leaf.jpg")
print(f"Diagnosis: {result.swahili}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Action: {result.action}")
```

### Teach the Model (Human-in-the-Loop)

```python
# When the prediction is wrong, correct it:
result = guard.teach(
    image_path="photo_of_leaf.jpg",
    true_label="northern_leaf_blight",
    confidence_weight=0.9,  # How sure are you?
)

# The model updates IMMEDIATELY — next prediction will be better
```

### Batch Processing

```python
# Process a whole folder of farmer photos
results = guard.batch_diagnose([
    "farmer_1.jpg", "farmer_2.jpg", "farmer_3.jpg",
])
for r in results:
    print(f"{r.swahili}: {r.confidence:.1%} — {r.action}")

# Export to CSV for record-keeping
csv_data = guard.batch_to_csv(results)
with open("diagnoses.csv", "w") as f:
    f.write(csv_data)
```

### Save and Resume

```python
# Save the trained model for next session
guard.save_model("models/session_2024.json")

# Later, resume from that save
restored_count = guard.load_model("models/session_2024.json")
print(f"Restored {restored_count} support images")
```

### System Health

```python
health = guard.system_health()
print(f"Calibration ECE: {health['calibration']['ece']}")
print(f"Session accuracy: {health['session']['accuracy']}")
print(f"Eco mode: {health['config']['eco_mode']}")
```

## Adapting MziziGuard for Your Own Use Case

### Add your own crops and diseases

Edit `config.yaml` to define new crops:

```yaml
crops:
  coffee:
    swahili: "kahawa"
    diseases:
      coffee_leaf_rust:
        swahili: "kutu ya majani ya kahawa"
        action: "Tumia dawa ya kuvu yenye shaba. Punguza kivuli."
        description: "Orange-yellow powdery spots on leaf undersides."
        severity: "high"
      healthy_coffee:
        swahili: "kahawa yenye afya"
        action: "Endelea na utunzaji wa kawaida."
        severity: "low"
```

### Load real images

Organize your photos like this and use `load_images_from_dir()`:

```
my_photos/
├── healthy_coffee/
│   ├── img_001.jpg
│   └── img_002.jpg
├── coffee_leaf_rust/
│   ├── img_003.jpg
│   └── img_004.jpg
└── coffee_berry_disease/
    └── img_005.jpg
```

### Other use cases that work with the same template

- **Coffee leaf rust detection** — major issue for Tanzania's cash crop
- **Cassava mosaic disease** — affects food security across East Africa
- **Poultry disease screening** — respiratory conditions from droppings photos
- **Skin condition triage** — community health workers in rural clinics
- **Manufacturing defect detection** — quality control for small industries

Any problem where a non-expert needs to classify images with just a few examples is a candidate for MziziGuard + AdaptShot.

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

## Project Structure

```
apps/tambua/
├── __init__.py        # Public API (MziziGuard, DiagnosisResult)
├── config.yaml         # Crop/disease definitions & engine settings
├── engine.py           # Core engine wrapping FewShotLearner
├── data.py             # Sample generation + real image folder loader
└── app.py              # Gradio web UI (5-tab interface)
```

## What This Project Demonstrates About AdaptShot

| Feature | How MziziGuard uses it |
|---------|----------------------|
| Few-shot learning | Only 5–10 images per disease class |
| CPU-only | Runs on any laptop, no GPU required |
| Offline capable | Full functionality with no internet |
| Human-in-the-loop | Teach tab corrects wrong predictions instantly |
| Calibrated confidence | Shows confidence % with every diagnosis |
| OOD detection | Flags non-crop images instead of guessing |
| Eco mode | `eco_mode: true` in config.yaml |
| Calibration report | Health tab shows ECE, temperature, metrics |
| Model persistence | `save_model()` / `load_model()` — resumes between sessions |
| Batch processing | Process entire folders of farmer photos at once |
| YAML configuration | Easy to add new crops/diseases without code changes |
| Swahili localization | Every diagnosis includes Swahili name + treatment advice |

---

## Next Steps

- Launch the full app: `tambua`
- Edit `config.yaml` to add your own crops and diseases
- Try loading real images with `load_images_from_dir()` instead of synthetic
- Present the terminal demo to non-technical audiences using `--no-pause`
- Check the [Full API Reference](../api/core.md) for all `FewShotLearner` methods

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
