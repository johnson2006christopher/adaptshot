# MziziGuard Complete Guide — Crop Disease Detection with AdaptShot

**A comprehensive, step-by-step reference for building, deploying, and extending MziziGuard.**

MziziGuard is a complete, working application that demonstrates AdaptShot's power: helping smallholder farmers identify crop diseases from just a handful of photos — no internet, no GPU, no expensive hardware. This guide covers everything from first launch to field deployment.

---

## Table of Contents

1. [What Is MziziGuard?](#what-is-mziziguard)
2. [Architecture Overview](#architecture-overview)
3. [Installation & First Launch](#installation-first-launch)
4. [The Web Application — Tab-by-Tab Walkthrough](#the-web-application-tab-by-tab-walkthrough)
5. [Python API — Complete Reference](#python-api-complete-reference)
6. [Configuration — config.yaml Deep Dive](#configuration-configyaml-deep-dive)
7. [Data Management](#data-management)
8. [Model Persistence — Save, Load, Resume](#model-persistence-save-load-resume)
9. [Batch Processing & Reporting](#batch-processing-reporting)
10. [Adapting MziziGuard for New Crops](#adapting-mziziguard-for-new-crops)
11. [The Terminal Demo](#the-terminal-demo)
12. [Field Deployment Guide](#field-deployment-guide)
13. [Troubleshooting MziziGuard](#troubleshooting-mziziguard)
14. [FAQ](#faq)

---

## What Is MziziGuard?

MziziGuard (Swahili: "root guardian") is a crop disease detection system powered by AdaptShot's few-shot vision learning engine. It enables:

| Capability | How |
|------------|-----|
| **Instant diagnosis** | Upload a photo → get disease name (Swahili + English), confidence score, and treatment advice |
| **Few-shot learning** | Train on as few as 5 photos per disease — no massive datasets needed |
| **Human-in-the-loop** | Correct wrong predictions; the model learns immediately, no retraining |
| **OOD detection** | Knows when it's out of its depth — won't guess on non-crop images |
| **Offline operation** | Works entirely without internet — built for the field |
| **CPU-only** | Runs on any laptop, no GPU required |
| **Swahili localization** | Disease names and treatment advice in Swahili |
| **Persistence** | Save/load trained models between sessions |

### The Problem It Solves

In Tanzania, 65% of the population depends on agriculture. Maize diseases like **Northern Leaf Blight** and **Gray Leaf Spot** destroy 20–60% of harvests every season. Agricultural extension officers can't reach every village. By the time a farmer gets a diagnosis, the crop is already lost.

Almost every farmer has a basic smartphone. MziziGuard turns that phone into a crop doctor — without needing internet in the field.

---

## Architecture Overview

```mermaid
graph TB
    A[Farmer takes photo] --> B[MziziGuard Web UI<br/>Gradio 5-tab interface]
    B --> C[MziziGuard Engine<br/>examples/mziziguard/engine.py]
    C --> D[AdaptShot FewShotLearner<br/>src/adaptshot/core/learner.py]
    D --> E[ResNet-18 Backbone<br/>Frozen feature extractor]
    E --> F[Embedding Vector]
    F --> G[Similarity Search<br/>Euclidean / Cosine]
    G --> H[CalibrationEngine<br/>Temperature Scaling]
    H --> I[ACTEngine<br/>Adaptive Threshold]
    I --> J[DiagnosisResult]
    J --> B
    B --> K[Farmer sees diagnosis<br/>Swahili name + action]
    K --> L{Wrong?}
    L -->|Yes| M[Extension Officer corrects]
    M --> N[learner.correct()]
    N --> O[CA-EWC Fine-tune]
    N --> P[UP-UGF Buffer]
    O --> D
    P --> D
    L -->|No| Q[Done]
```

### Project Structure

```
examples/mziziguard/
├── __init__.py        # Public API: MziziGuard, DiagnosisResult, DiseaseInfo
├── config.yaml         # Crop and disease definitions
├── engine.py           # Core engine (543 lines) wrapping FewShotLearner
├── data.py             # Sample generation + folder-based image loading
└── app.py              # Gradio web UI — 5-tab interface (562 lines)
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI | Gradio 6.x | Web interface with 5 interactive tabs |
| Engine | Python 3.9+ | Core application logic |
| ML Backend | AdaptShot (PyTorch) | Few-shot learning, calibration, ACT |
| Config | YAML (PyYAML) | Crop definitions, engine settings |
| Images | PIL/Pillow | Synthetic image generation, preprocessing |

---

## Installation & First Launch

### Prerequisites

- Python 3.9 or newer
- 250MB+ free RAM
- Internet for first `pip install` only (backbone weights ~45MB cached after)

### Step 1: Install AdaptShot with UI Support

```bash
# From PyPI
pip install "adaptshot[ui]"

# Or from source (recommended for development)
git clone https://github.com/johnson2006christopher/adaptshot.git
cd adaptshot
pip install -e ".[ui]"
```

What `[ui]` installs:
- `adaptshot` — core few-shot learning engine
- `gradio>=3.50.0` — web UI framework
- `PyYAML>=6.0` — configuration parsing

### Step 2: Launch the Web Application

```bash
# Default: http://localhost:7860
python -m examples.mziziguard.app

# Custom port
python -m examples.mziziguard.app --port 8080

# Public shareable link (for demos)
python -m examples.mziziguard.app --share
```

### Step 3: Verify

Open your browser to `http://localhost:7860`. You should see:

- 🌽 **Header**: "MziziGuard — Crop Disease Detection"
- ⚙️ **Setup tab**: Options to generate samples or load real images
- 🔍 **Diagnose tab**: Image upload and predict button
- 👩‍🏫 **Teach tab**: Correction interface
- 🏥 **Health tab**: System metrics dashboard
- 📦 **Batch tab**: Multi-image processing

---

## The Web Application — Tab-by-Tab Walkthrough

### Tab 1: ⚙️ Setup — Train the Model

This is where you give MziziGuard its "knowledge" — the reference photos it uses to diagnose new images.

**Option A: Generate Synthetic Samples (Quick Start)**

1. Use the slider to choose how many images per class (default: 5)
2. Click **"Generate Samples & Train"**
3. The system creates synthetic leaf images using PIL and loads them into AdaptShot
4. Status message confirms training: `"Generated 15 support images across 3 classes"`

**Option B: Load Real Images**

Organize your photos in a folder structure:

```
my_crop_photos/
├── healthy_maize/
│   ├── leaf_001.jpg
│   ├── leaf_002.jpg
│   └── leaf_003.jpg
├── northern_leaf_blight/
│   ├── blight_001.jpg
│   └── blight_002.jpg
└── gray_leaf_spot/
    └── spot_001.jpg
```

Then:
1. Enter the folder path in the text box
2. Optionally set a max images per class limit
3. Click **"Load from Folder"**

### Tab 2: 🔍 Diagnose — Identify Crop Diseases

**Step-by-step flow:**

1. **Upload** a crop photo (drag-and-drop or click to browse)
2. Click **"Diagnose"**
3. Results appear immediately:

| Output | Description |
|--------|-------------|
| Diagnosis (Swahili) | Disease name in Swahili with severity emoji |
| Confidence | Percentage bar showing how sure the model is |
| Recommended Action | Treatment advice (e.g., "Apply fungicide. Rotate crops.") |

**Behind the scenes** (what AdaptShot does):
1. ResNet-18 extracts an embedding vector from the uploaded photo
2. Compares it to all stored support embeddings via Euclidean distance
3. Finds the nearest prototype (prototypical inference mode)
4. Temperature-scales the raw similarity into calibrated confidence
5. ACT engine decides whether to accept or request feedback
6. OOD detector checks if the image is too far from known classes

### Tab 3: 👩‍🏫 Teach — Human-in-the-Loop Correction

This is AdaptShot's most powerful feature. When the model makes a mistake, you correct it — and it learns immediately.

**How to use:**

1. First, make a prediction in the **Diagnose** tab
2. If it's wrong, switch to the **Teach** tab
3. Select the **correct label** from the dropdown (or type a new one)
4. Set your **confidence** in your correction (0.0 = unsure, 1.0 = completely sure)
5. Click **"Submit Correction"**

**What happens:**

- The correction flows into AdaptShot's `correct()` pipeline
- Calibration engine updates temperature scaling
- ACT thresholds adjust per-class
- If enough corrections accumulate, CA-EWC fine-tunes the classification head
- Example is added to the replay buffer
- **Every correction makes the model smarter for the next farmer**

### Tab 4: 🏥 System Health — Calibration Dashboard

Monitor how well your MziziGuard is performing:

| Metric | What It Means |
|--------|--------------|
| **ECE** (Expected Calibration Error) | How well confidence scores match actual accuracy. Lower is better. |
| **Debiased ECE** | ECE corrected for finite-sample bias |
| **Temperature** | Current calibration temperature (1.0 = no scaling) |
| **Window Size** | Number of corrections in the calibration window |
| **OOD Threshold** | Distance beyond which images are flagged as out-of-distribution |
| **Support Size** | Total number of stored support/correction embeddings |
| **Prototype Count** | Number of class prototypes computed |

**Session stats:**

- Total predictions made
- Total corrections submitted
- Accuracy: `(predictions - corrections) / predictions`
- Session duration

### Tab 5: 📦 Batch — Process Multiple Images

Upload multiple photos at once and get a summary table:

1. Click **"Upload Crop Photos"** and select multiple images
2. Click **"Batch Diagnose"**
3. Results table shows:

| # | Image | Diagnosis (Swahili) | Confidence | Severity | Action |
|---|-------|---------------------|------------|----------|--------|

Use this for extension officers processing photos from many farmers, or for testing the model against a set of known images.

---

## Python API — Complete Reference

MziziGuard exposes a clean Python API for programmatic use, scripting, and integration.

### Initialization

```python
from examples.mziziguard import MziziGuard

# Load with default config (looks for config.yaml next to engine.py)
guard = MziziGuard()

# Or specify a custom config path
guard = MziziGuard("path/to/my_config.yaml")
```

### Core Methods

#### `initialize_with_samples(n_support=5, data_dir=None, seed=42) -> int`

Generate synthetic training images and load them into the model.

```python
count = guard.initialize_with_samples(n_support=5, seed=42)
print(f"Loaded {count} support images")  # 15 = 3 classes × 5 images
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_support` | `int` | `5` | Images per disease class |
| `data_dir` | `str` or `None` | `None` | Where to write images (temp dir if None) |
| `seed` | `int` | `42` | Reproducibility seed |

#### `load_images_from_dir(image_dir, max_per_class=0) -> int`

Load real images from a folder-per-class directory structure.

```python
count = guard.load_images_from_dir("data/crop_photos/", max_per_class=10)
print(f"Loaded {count} images from directory")
```

#### `diagnose(image) -> DiagnosisResult`

Run disease diagnosis on a single image.

```python
result = guard.diagnose("photo.jpg")

# Human-readable output
print(f"Diagnosis: {result.swahili}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Action: {result.action}")
print(f"Severity: {result.severity}")
print(f"OOD: {result.ood_flag}")
```

**DiagnosisResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | English disease name |
| `swahili` | `str` | Swahili disease name |
| `confidence` | `float` | Calibrated confidence (0.0–1.0) |
| `raw_confidence` | `float` | Uncalibrated raw confidence |
| `action` | `str` | Treatment advice |
| `severity` | `str` | `low`, `moderate`, `high`, `critical`, `unknown` |
| `ood_flag` | `bool` | `True` if image is out-of-distribution |
| `uncertainty_flag` | `bool` | `True` if model is uncertain |
| `act_action` | `str` | ACT engine decision (`ACCEPT`, `REQUEST_FEEDBACK`, etc.) |
| `distance_to_prototype` | `float` | Distance to nearest class prototype |
| `calibrated_ece` | `float` | Debiased ECE at time of prediction |

#### `teach(image_path, true_label, confidence_weight=1.0) -> Dict[str, Any]`

Correct a wrong prediction — this is the human-in-the-loop magic.

```python
result = guard.teach(
    image_path="photo.jpg",
    true_label="northern_leaf_blight",
    confidence_weight=0.9,
)
print(f"Fine-tuned: {result['fine_tuned']}")
print(f"Buffer size: {result['buffer_size']}")
```

#### `teach_from_ui(true_label, confidence_weight=1.0) -> str`

Convenience method for Gradio UI (uses last predicted image automatically).

```python
status = guard.teach_from_ui(
    true_label="gray_leaf_spot",
    confidence_weight=0.8,
)
# → "✅ Correction recorded! Fine-tuned: False, Buffer: 1"
```

#### `batch_diagnose(image_paths) -> List[DiagnosisResult]`

Process multiple images at once.

```python
results = guard.batch_diagnose([
    "farmer_1.jpg",
    "farmer_2.jpg",
    "farmer_3.jpg",
])
for r in results:
    print(f"{r.swahili}: {r.confidence:.1%}")
```

#### `batch_to_csv(results) -> str`

Convert batch results to CSV format.

```python
csv = guard.batch_to_csv(results)
with open("diagnoses.csv", "w") as f:
    f.write(csv)
```

#### `system_health() -> Dict[str, Any]`

Get calibration metrics, session stats, and config summary.

```python
health = guard.system_health()
print(health["calibration"]["ece"])          # Expected Calibration Error
print(health["session"]["accuracy"])          # Session accuracy
print(health["config"]["eco_mode"])           # Eco mode enabled?
```

#### `save_model(path) -> str`

Save the trained model to disk (three files: `.json`, `.embeddings.npy`, `.head.pt`).

```python
guard.save_model("models/session_2024.json")
```

#### `load_model(path) -> int`

Restore a saved model.

```python
count = guard.load_model("models/session_2024.json")
print(f"Restored {count} support images")
```

#### `label_to_info(label) -> DiseaseInfo`

Get structured info for any disease label.

```python
info = guard.label_to_info("northern_leaf_blight")
print(info.swahili)   # "ugonjwa wa mabaka ya kahawia"
print(info.action)     # Treatment advice
print(info.severity)   # "moderate"
```

### Complete Workflow Example

```python
from examples.mziziguard import MziziGuard

# 1. Initialize
guard = MziziGuard()
guard.initialize_with_samples(n_support=5, seed=42)

# 2. Diagnose
result = guard.diagnose("field_photo.jpg")
print(f"DIAGNOSIS: {result.swahili}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Action: {result.action}")

# 3. Correct if wrong
if result.label != "northern_leaf_blight":
    guard.teach("field_photo.jpg", "northern_leaf_blight", confidence_weight=0.9)
    print("✓ Model corrected — next prediction will be better")

# 4. Check health
health = guard.system_health()
print(f"ECE: {health['calibration']['ece']}")
print(f"Session accuracy: {health['session']['accuracy']}")

# 5. Save for next session
guard.save_model("models/session_2024.json")
print("✓ Model saved for next session")
```

---

## Configuration — config.yaml Deep Dive

The `config.yaml` file is the single source of truth for MziziGuard's behavior. You should edit this file (not code) to add crops, change engine settings, or modify treatment advice.

### Application Metadata

```yaml
application:
  name: "MziziGuard"
  version: "0.1.0"
  description: "Crop disease detection for smallholder farmers"
```

### Engine Settings

```yaml
engine:
  backbone: "resnet18"           # resnet18 (more accurate) | mobilenet_v3_small (faster/lighter)
  device: "cpu"                  # cpu (recommended) | cuda | mps
  seed: 42                       # Reproducibility seed
  inference_mode: "prototypical" # prototypical (best for few-shot) | nearest_neighbor
  similarity_metric: "euclidean" # euclidean | cosine
  eco_mode: true                 # Carbon-aware inference — saves battery in the field
  enable_ood_detection: true     # Catch non-crop images
```

### Crop & Disease Definitions

Each crop has one or more diseases with Swahili names, treatment actions, and severity:

```yaml
crops:
  maize:
    swahili: "mahindi"
    diseases:
      healthy_maize:
        swahili: "mahindi yenye afya"
        action: "Hakuna matibabu yanayohitajika."
        description: "Healthy maize with no visible disease symptoms."
        severity: "low"
      northern_leaf_blight:
        swahili: "ugonjwa wa mabaka ya kahawia"
        action: "Ondoa majani yaliyoathirika. Tumia dawa ya kuvu."
        description: "Cigar-shaped lesions caused by Exserohilum turcicum."
        severity: "moderate"
```

### Adding a New Crop

```yaml
crops:
  maize:
    # ... existing maize config ...

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

### Localization

```yaml
localization:
  language: "sw"    # sw = Swahili, en = English
  fallback: "en"    # Fallback language if translation missing
```

### Paths

```yaml
paths:
  model_dir: "models"      # Where save_model() writes to
  sample_data: "samples"   # Sample image cache directory
```

---

## Data Management

### Loading Real Images

MziziGuard supports ImageFolder-style directory loading. Organize photos like this:

```
data/crop_photos/
├── healthy_maize/
│   ├── leaf_001.jpg
│   ├── leaf_002.jpg
│   └── leaf_003.jpg
├── northern_leaf_blight/
│   ├── blight_001.jpg
│   └── blight_002.jpg
└── gray_leaf_spot/
    └── spot_001.jpg
```

```python
guard.load_images_from_dir("data/crop_photos/", max_per_class=10)
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`.

### Generating Synthetic Samples

For quick testing without real images:

```python
from examples.mziziguard.data import generate_samples

support_paths, support_labels, query_paths, query_labels = generate_samples(
    output_dir="/tmp/samples",
    n_support=5,  # 5 images per class for training
    n_query=3,    # 3 images per class for testing
    seed=42,
)
```

The synthetic generators create:
- **healthy_maize**: Green oval with veins on soil background
- **northern_leaf_blight**: Healthy leaf + cigar-shaped brown lesions
- **gray_leaf_spot**: Healthy leaf + rectangular gray spots

### Programmatic Image Loading

```python
from examples.mziziguard.data import load_from_folders, list_classes_from_dir

# List available classes
classes = list_classes_from_dir("data/photos/")
print(f"Found classes: {classes}")

# Load with limits
paths, labels = load_from_folders("data/photos/", max_per_class=10)
```

---

## Model Persistence — Save, Load, Resume

MziziGuard leverages AdaptShot's built-in checkpointing, which saves everything needed to resume a session:

### Saving

```python
guard.save_model("models/my_session.json")
```

Creates three files:

| File | Contents |
|------|----------|
| `models/my_session.json` | Configuration, calibration history, ACT thresholds, buffer metadata, label index |
| `models/my_session.embeddings.npy` | NumPy array of all support/correction embedding vectors |
| `models/my_session.head.pt` | PyTorch state dict for the fine-tuned classification head |

### Loading

```python
guard = MziziGuard()
count = guard.load_model("models/my_session.json")
print(f"Restored {count} support images from previous session")
```

### Typical Workflow with Persistence

```python
from pathlib import Path
import json
from examples.mziziguard import MziziGuard

SESSION_FILE = "models/latest.json"
guard = MziziGuard()

# Try to resume from last session
if Path(SESSION_FILE).exists():
    count = guard.load_model(SESSION_FILE)
    print(f"Resumed session with {count} images")
else:
    # First time — train fresh
    guard.initialize_with_samples(n_support=5)
    print("Fresh training complete")

# ... do work ...

# Save progress
guard.save_model(SESSION_FILE)
```

---

## Batch Processing & Reporting

### Processing Multiple Images

```python
guard = MziziGuard()
guard.initialize_with_samples(n_support=5)

# Process a folder
from pathlib import Path
photos = list(Path("field_photos/").glob("*.jpg"))
results = guard.batch_diagnose([str(p) for p in photos])

# Print summary
for path, result in zip(photos, results):
    print(f"{path.name}: {result.swahili} ({result.confidence:.1%})")
```

### CSV Export

```python
csv_data = guard.batch_to_csv(results)
with open("diagnoses.csv", "w", encoding="utf-8") as f:
    f.write(csv_data)
```

### Programmatic Report Generation

```python
health = guard.system_health()

report = f"""
=== MziziGuard System Report ===

Calibration:
  ECE: {health['calibration']['ece']}
  Temperature: {health['calibration']['temperature']}
  Window Size: {health['calibration']['window_size']}
  Support Size: {health['calibration']['support_size']}

Session:
  Predictions: {health['session']['total_predictions']}
  Corrections: {health['session']['total_corrections']}
  Accuracy: {health['session']['accuracy']:.1%}

Config:
  Backbone: {health['config']['backbone']}
  Device: {health['config']['device']}
  Eco Mode: {health['config']['eco_mode']}
"""
print(report)
```

---

## Adapting MziziGuard for New Crops

### Step 1: Edit config.yaml

Add your crop and diseases to `examples/mziziguard/config.yaml`:

```yaml
crops:
  cassava:
    swahili: "muhogo"
    diseases:
      healthy_cassava:
        swahili: "muhogo wenye afya"
        action: "Endelea na utunzaji wa kawaida."
        severity: "low"
      cassava_mosaic:
        swahili: "ugonjwa wa mosai ya muhogo"
        action: "Ondoa mimea iliyoathirika. Tumia vipando sugu."
        description: "Yellow-green mosaic patterns on leaves."
        severity: "high"
      cassava_brown_streak:
        swahili: "ugonjwa wa mistari ya kahawia"
        action: "Ondoa mimea yote iliyoathirika. Panda aina sugu."
        severity: "critical"
```

### Step 2: Prepare Training Images

Organize photos by class:

```
data/cassava/
├── healthy_cassava/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   ├── img_003.jpg
│   ├── img_004.jpg
│   └── img_005.jpg
├── cassava_mosaic/
│   ├── img_006.jpg
│   └── ... 5 images
└── cassava_brown_streak/
    └── ... 5 images
```

### Step 3: Train

```python
guard = MziziGuard()
guard.load_images_from_dir("data/cassava/", max_per_class=5)
# Now ready to diagnose cassava diseases
```

### Step 4: Launch the App

```bash
python -m examples.mziziguard.app
```

The new diseases will automatically appear in the dropdown and UI.

### Other Use Cases

The MziziGuard template adapts easily to:

- **Coffee leaf rust** — Tanzania's major cash crop
- **Cassava mosaic/brown streak** — food security across East Africa
- **Banana bacterial wilt** — staple food crop
- **Tomato blight** — market garden crops
- **Rice blast** — wetland farming
- **Poultry disease screening** — respiratory conditions from droppings
- **Skin condition triage** — community health workers
- **Manufacturing QA** — visual defect detection

---

## The Terminal Demo

For presentations to non-technical audiences, the terminal demo walks through 6 narrated stages:

```bash
# Interactive (press Enter between stages)
python examples/crop_disease_demo.py

# Non-interactive (for testing)
python examples/crop_disease_demo.py --no-pause
```

**Stages:**

| Stage | Title | Concept Demonstrated |
|-------|-------|---------------------|
| 0 | Why This Matters | Problem framing |
| 1 | Learning from 5 Photos | Few-shot learning |
| 2 | Farmer Takes a Photo | Inference & prediction |
| 3 | Human Teaches Machine | Human-in-the-loop |
| 4 | "I Don't Know" | OOD detection |
| 5 | System Health Report | Calibration monitoring |
| 6 | Why AdaptShot, Why Tanzania | Vision & mission |

---

## Field Deployment Guide

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any x86-64, 2015+ | Intel Core i3 / AMD Ryzen, 2018+ |
| RAM | 250MB free | 512MB+ free |
| Storage | 100MB for app + 45MB for backbone | 500MB+ with image storage |
| OS | Linux, Windows, macOS | Ubuntu 20.04+ / Windows 10+ |
| Internet | First install only | None needed after setup |

### Deployment Checklist

1. **Install AdaptShot** on the target machine
2. **Copy config.yaml** with custom crop definitions
3. **Prepare training images** (5–10 per disease class)
4. **Train the model** via the Setup tab
5. **Save the model** (`guard.save_model()`) for quick restart
6. **Launch the app** (`python -m examples.mziziguard.app`)
7. **Bookmark** `http://localhost:7860` in the browser
8. **Test** with known images before field use

### Offline Operation

After initial installation:
- No internet required for predictions
- No internet required for corrections
- No internet required for model saving/loading
- Backbone weights cached at `~/.cache/torch/`
- All embeddings stored in RAM

### Multi-User Setup

For an extension office with multiple officers:

```bash
# Each officer gets their own port
python -m examples.mziziguard.app --port 7861  # Officer A
python -m examples.mziziguard.app --port 7862  # Officer B
```

Each instance maintains independent state. Corrections from one don't affect the other unless you share model checkpoints.

---

## Troubleshooting MziziGuard

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'gradio'` | Gradio not installed | `pip install "adaptshot[ui]"` |
| `ModuleNotFoundError: No module named 'yaml'` | PyYAML not installed | `pip install PyYAML` |
| App starts but shows "Not trained" | No support images loaded | Go to Setup tab → Generate Samples |
| All predictions are "healthy_maize" | Too few support images | Increase n_support to 5–10 per class |
| Confidence is always 100% | Calibration not warmed up | Make 10+ predictions; correct some |
| OOD flag never triggers | OOD detection not enabled | Set `enable_ood_detection: true` in config |
| App crashes on startup | Port 7860 in use | Use `--port 8080` |
| "Model not trained yet" error | Skipped Setup tab | Train model first in Setup tab |
| Gradio deprecation warnings | Gradio 6.x API changes | Update to MziziGuard v0.1.0+ |

### Verification Commands

```bash
# Check Python version
python --version  # Should be 3.9+

# Check AdaptShot import
python -c "from adaptshot import FewShotLearner; print('OK')"

# Check MziziGuard import
python -c "from examples.mziziguard import MziziGuard; print('OK')"

# Quick smoke test
python -c "
from examples.mziziguard import MziziGuard
guard = MziziGuard()
guard.initialize_with_samples(n_support=3)
print(f'Trained: {guard.is_trained}, Classes: {guard.known_labels}')
"
```

---

## FAQ

**Q: Can I use MziziGuard without any coding?**

A: Yes! Launch `python -m examples.mziziguard.app` and use the web interface. No coding required after installation.

**Q: How many photos per disease do I need?**

A: As few as 3–5 per class. More is better (10–20 ideal), but AdaptShot's few-shot learning is designed to work with very small support sets.

**Q: Does it work with real farm photos?**

A: Yes. Use the "Load from Folder" option in the Setup tab, or `load_images_from_dir()` in the API.

**Q: Can I add my own diseases?**

A: Yes. Edit `config.yaml` to add disease definitions. No code changes needed.

**Q: Does it require internet in the field?**

A: No. After installation, MziziGuard runs fully offline. Perfect for rural areas.

**Q: Can multiple people use it at once?**

A: Launch multiple instances on different ports. Each is independent.

**Q: How accurate is it?**

A: With 5+ support images per class, accuracy can reach 80–95% depending on the visual distinctiveness of diseases. Human correction continuously improves accuracy.

**Q: What if someone shows it a photo of something else?**

A: The OOD (out-of-distribution) detector flags non-crop images instead of guessing. This is critical for trust in the field.

**Q: Can I export results?**

A: Yes. Use the Batch tab to process multiple images and copy the table. Or use `batch_to_csv()` in the API.

**Q: How do I update from the terminal demo to the full app?**

A: The terminal demo (`crop_disease_demo.py`) is for presentations. The full app (`python -m examples.mziziguard.app`) is for real use. Both use the same engine.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
