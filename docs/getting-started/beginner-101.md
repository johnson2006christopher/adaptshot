# Beginner 101: Your First AdaptShot Prediction

Welcome! This guide assumes you know Python basics and nothing about AI. By the end, you will have trained a model with 6 images, made a prediction with uncertainty quantification, and corrected it when it was wrong — all on a normal laptop with no internet required after installation.

## What Is Few-Shot Learning?

Most AI needs thousands or millions of labeled images to learn. Few-shot learning is different: it learns from **fewer than 50 examples per class**. AdaptShot does this by:

1. Using a pre-trained "backbone" (ResNet-18 or MobileNetV3) to convert images into numbers (embeddings)
2. Comparing new images to the ones it has already seen
3. Finding the closest match — and measuring how uncertain it is about that match

Think of it like a detective: instead of needing to see every possible leaf to identify crop disease, it compares your new photo to the few reference photos you gave it. And crucially, it tells you when it's not sure.

## What Does AdaptShot Do?

AdaptShot is a Python library that lets you:

- **Classify images** with only a handful of examples per category
- **Know when it's uncertain** — via three different uncertainty signals that combine to give honest confidence
- **Provide prediction sets** instead of single answers — e.g., "It's either cat or dog" with 95% guaranteed coverage
- **Learn from corrections** — every time you correct it, it gets better without forgetting what it already knew
- **Explain its decisions** — showing which reference images influenced the prediction and why
- **Run entirely on CPU** — no expensive GPU required
- **Work offline** — no internet needed after the first setup

### Concrete Use Cases

| Domain | Example | Why AdaptShot Fits |
|--------|---------|-------------------|
| Agriculture | Identify cassava mosaic virus from 10 leaf photos | Works offline in the field; admits uncertainty on rare variants |
| Healthcare | Triage pneumonia vs. normal from chest X-rays | Calibrated confidence prevents dangerous overconfidence |
| Conservation | Classify camera trap images of endangered species | Runs on battery-powered Raspberry Pi in remote areas |
| Quality Control | Detect manufacturing defects from a few reference images | Learns new defect types from human corrections |

## Prerequisites

You need:

- Python 3.9 or newer (`python --version` to check)
- At least 250MB of free RAM
- Internet for the first `pip install` only
- A computer — any laptop from the last 5 years works

No GPU needed. No cloud account needed. No large datasets needed.

## Step 1: Install AdaptShot

Open your terminal and run:

```bash
pip install adaptshot
```

Verify it worked:

```bash
python -c "from adaptshot import FewShotLearner; print('Installed!')"
```

Expected output:

```text
Installed!
```

## Step 2: Create a Beginner Script

Create a file called `my_first_adaptshot.py` and paste this:

```python
"""Your first AdaptShot prediction — no external images needed."""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from adaptshot import FewShotLearner
from adaptshot.config.settings import AdaptShotConfig


# Step 1: Create 6 synthetic images (3 "cat", 3 "dog")
def make_image(path: Path, color: tuple[int, int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :] = np.array(color, dtype=np.uint8)
    noise = rng.integers(0, 30, size=(224, 224, 3), dtype=np.uint8)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


with tempfile.TemporaryDirectory(prefix="adaptshot_beginner_") as tmp:
    root = Path(tmp)
    cat_dir = root / "cat"
    dog_dir = root / "dog"
    cat_dir.mkdir()
    dog_dir.mkdir()

    image_paths: list[str] = []
    labels: list[str] = []

    # Create 3 "cat" images (green-ish)
    for i in range(3):
        path = cat_dir / f"cat_{i}.png"
        make_image(path, color=(40, 150, 50), seed=i)
        image_paths.append(str(path))
        labels.append("cat")

    # Create 3 "dog" images (brown-ish)
    for i in range(3):
        path = dog_dir / f"dog_{i}.png"
        make_image(path, color=(150, 80, 35), seed=100 + i)
        image_paths.append(str(path))
        labels.append("dog")

    # Create a query image (looks like a dog)
    query_path = root / "mystery_animal.png"
    make_image(query_path, color=(150, 80, 35), seed=999)

    # Step 2: Configure the learner (all CPU, no GPU needed)
    config = AdaptShotConfig(
        backbone="resnet18",
        device="cpu",
        seed=42,
        max_buffer_size=20,
        verbose=True,
    )
    learner = FewShotLearner(config=config)

    # Step 3: Load the support images (what the model learns from)
    print("Loading support set...")
    learner.load_support_images(image_paths=image_paths, labels=labels)
    print(f"Loaded {len(image_paths)} images across {len(set(labels))} classes")

    # Step 4: Make a prediction
    print(f"\nPredicting: {query_path.name}...")
    result = learner.predict(str(query_path))

    print(f"  Prediction: {result.prediction}")
    print(f"  Confidence: {result.calibrated_confidence:.1%}")
    print(f"  Needs human review: {result.uncertainty_flag}")
    print(f"  ACT action: {result.act_action}")

    # v0.2.0: Conformal prediction set
    if result.conformal_set:
        print(f"  Conformal set (90% coverage): {result.conformal_set}")

    # v0.2.0: Uncertainty signals
    if result.uncertainty_report:
        print(f"  Epistemic uncertainty: {result.uncertainty_report.get('epistemic', 0):.3f}")
        print(f"  Aleatoric uncertainty: {result.uncertainty_report.get('aleatoric', 0):.3f}")

    # Step 5: If it got it wrong, correct it
    true_label = "dog"
    if str(result.prediction) != true_label:
        print(f"\nCorrecting: model predicted '{result.prediction}', "
              f"true label is '{true_label}'")
        feedback = learner.correct(
            image_path=str(query_path),
            true_label=true_label,
            confidence_weight=0.9,
        )
        print(f"  Correction routed: {feedback['calibration_updated']}")
        print(f"  Buffer size after correction: {feedback['buffer_size']}")

    print("\nDone! You just ran your first AdaptShot pipeline.")
```

Save the file and run it:

```bash
python my_first_adaptshot.py
```

Expected output:

```text
Loading support set...
Loaded 6 images across 2 classes

Predicting: mystery_animal.png...
  Prediction: dog
  Confidence: 99.9%
  Needs human review: False
  ACT action: ACCEPT
  Conformal set (90% coverage): ['dog', 'cat']
  Epistemic uncertainty: 0.031
  Aleatoric uncertainty: 0.089

Done! You just ran your first AdaptShot pipeline.
```

## What Just Happened?

1. **`AdaptShotConfig`**: You told the library to use ResNet-18 on CPU with seed 42 (deterministic)
2. **`FewShotLearner(config=config)`**: Created a learner ready to classify images
3. **`load_support_images(paths, labels)`**: Gave the learner 6 reference images (3 per class)
4. **`predict(query)`**: The learner:
   - Extracted an embedding from the query image
   - Compared it to all support examples
   - Computed calibrated confidence via temperature scaling
   - Generated a conformal prediction set (guaranteed to contain the true class 90% of the time)
   - Measured epistemic and aleatoric uncertainty
5. **`correct(path, label, weight)`**: (If needed) Corrected a wrong prediction so the learner improves

## Understanding The Output

| Field | Meaning |
|-------|---------|
| `prediction` | The class the model thinks is most likely |
| `calibrated_confidence` | How sure the model is (0% to 100%), adjusted for honesty |
| `uncertainty_flag` | `True` means "don't trust me, get a human" |
| `act_action` | `ACCEPT` means the model is confident; `REQUEST_FEEDBACK` means it wants help |
| `conformal_set` | Set of classes guaranteed to contain the true answer (at configured coverage level) |
| `epistemic uncertainty` | How much the model's knowledge is unstable — high means "I need more training data" |
| `aleatoric uncertainty` | How much noise is in the data itself — high means "this is a genuinely ambiguous case" |

## What About The GUI?

If you prefer clicking instead of coding, AdaptShot has two browser-based interfaces:

**Gradio Dashboard** (simpler):
```bash
pip install "adaptshot[ui]"
python -m adaptshot.ui.app
```

**AdaptShot Studio** (full-featured):
```bash
pip install "adaptshot[gui]"
adaptshot-studio
```

Both run in your browser at `http://127.0.0.1:7860`. The underlying logic is the same `FewShotLearner` you just used.

## Next Steps

Now that you understand the basics:

- **[Quick Start](quickstart.md)**: Run the official smoke test with latency and memory profiling
- **[Installation](installation.md)**: Learn about FAISS acceleration, ONNX deployment, and optional extras
- **[Tutorial 1: Getting Started](../tutorials/01_getting_started.md)**: Deeper dive into the API
- **[Tutorial 2: Human in the Loop](../tutorials/02_human_in_the_loop.md)**: Master corrections and continual learning
- **[Memory Profiling](../tutorials/13_profiling_memory.md)**: Monitor RAM usage at every lifecycle stage

---

## Join the Community

AdaptShot is built by and for a global community. Here's how to get involved:

- **[⭐ Star on GitHub](https://github.com/johnson2006christopher/adaptshot)** — Every star helps us reach more developers who need CPU-first AI
- **[📱 Join WhatsApp](https://chat.whatsapp.com/J6AbrvbjmBc5XXX2fnN6RK)** — Ask questions, share projects, collaborate in real time
- **[💬 GitHub Discussions](https://github.com/johnson2006christopher/adaptshot/discussions)** — Propose features, share use cases, get help
- **[🔧 Contribute](../contributing.md)** — Write code, improve docs, or report bugs

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'adaptshot'` | Run `pip install adaptshot` |
| `ImportError: cannot import name 'FewShotLearner'` | Check your Python version: `python --version` (needs 3.9+) |
| `ConfigValidationError: device must be 'cpu'` | AdaptShot is CPU-first; set `device="cpu"` unless you have CUDA configured |
| `InvalidImageError` | Make sure your images are RGB PNG or JPEG files |
| Script runs slowly first time | The backbone downloads on first use (~45MB); cached after that |
| `conformal_set` is empty | Increase `conformal_alpha` (e.g., 0.10 for 90% coverage); very small alpha may produce empty sets on small support sets |
