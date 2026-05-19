---
title: "01 — Getting Started with AdaptShot (v0.1.1)"
nav_order: 1
---

!!! note
    This tutorial assumes you know basic Python (variables, lists, functions) but are new to machine learning. Every technical term is explained with a simple real-world analogy.

**What is AdaptShot?**

AdaptShot is a tiny toolkit that lets a program learn from just a few example images and then make predictions about new images. Think of it like teaching a friend to recognize plants from 10 photos instead of training them with 10,000 photos — fast and lightweight.

**Why CPU / Offline / Low‑Resource?**

AdaptShot is designed to run on ordinary laptops and low-power devices (CPU-only). This keeps costs, electricity use, and carbon emissions low — important if you want to run models where power or internet are limited (for example, in remote or under-resourced areas). In short: it’s built to be fair and practical.

!!! warning
    AdaptShot v0.1.1 is CPU-first. The library validates `device == "cpu"` and will raise an error otherwise. See `src/adaptshot/core/learner.py` for details.

**Installation**

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install AdaptShot in editable mode (run from the repository root):

```bash
pip install -e .[dev]
```

3. Quick verify: start Python and import the main learner class:

```bash
python -c "import sys, os; sys.path.insert(0, os.path.join(os.getcwd(), 'src')); from adaptshot.core.learner import FewShotLearner; print('AdaptShot import OK')"
# Expected output: AdaptShot import OK
```

!!! tip
    If `import` fails, ensure you ran the command from the repository root and that `src/` exists. Adding `src` to `PYTHONPATH` or using the editable install above fixes this.

**Your First Prediction — copy, paste, run**

Read → Type → Run → See Output → Understand Why.

Create a new file called `first_predict.py` at the repository root and paste this exact script. It is 8 lines (no extra blank lines) and is runnable from the repo root. The script creates tiny example images on disk, loads them as the support set, runs a prediction, and prints the structured result.

```python
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from adaptshot.core.learner import FewShotLearner
from PIL import Image
paths=[(Image.new("RGB",(32,32),(i*30,i*30,i*30)).save(p:=(f"tmp_support_{i}.png")) or p) for i in range(3)]
labels=["leaf","bark","leaf"]
learner=FewShotLearner()  # constructor: see src/adaptshot/core/learner.py
learner.load_support_images(paths, labels)  # loads 3 example images (see load_support_images in src/adaptshot/core/learner.py)
print(learner.predict(paths[0]))  # run prediction and print PredictionResult (see predict in src/adaptshot/core/learner.py)
# Expected output: a PredictionResult object showing a `prediction` and confidence fields
```

**Line-by-Line Explanation**

- `import os, sys` — bring in basic tools to manipulate paths (like telling Python where to look for code). Analogy: pointing your friend to the right cookbook on a shelf.
- `sys.path.insert(0, os.path.join(os.getcwd(), "src"))` — ensure Python can find the `adaptshot` package inside the `src/` folder. If you installed via `pip install -e .[dev]` this is optional. See `src/adaptshot/core/learner.py` for the learner implementation.
- `from adaptshot.core.learner import FewShotLearner` — imports the main class you use to teach and query the model. (Reference: [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py))
- `from PIL import Image` — we use Pillow (a lightweight image library) to create small example images. Analogy: drawing quick sketches to show your friend what to look for.
- `paths=[(...)]` — creates and saves three tiny RGB images to disk (`tmp_support_0.png`, `tmp_support_1.png`, `tmp_support_2.png`). These image files act as the “support set” (examples you teach from).
- `labels=["leaf","bark","leaf"]` — a plain Python list of labels matching each support image. Analogy: sticky notes attached to each photo saying what it is.
- `learner=FewShotLearner()` — create the learner object that holds the examples, similarity buffers, and simple decision logic. See the class in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `learner.load_support_images(paths, labels)` — load the three images and compute internal embeddings used for similarity lookups. This is how you “teach” AdaptShot. See `load_support_images` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- `learner.predict(paths[0])` — get a prediction for the provided image path. The method returns a `PredictionResult` object with fields `prediction`, `raw_confidence`, `calibrated_confidence`, `neighbor_idx`, `uncertainty_flag`, and `act_action`. See `predict` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).

**Why this works (plain language)**

- AdaptShot converts each photo into a numeric fingerprint (an "embedding") that captures visual patterns. Analogy: turning a photo into a short checklist of important features.
- To predict, AdaptShot compares the new photo's fingerprint to the stored examples and picks the nearest one (nearest neighbor). Analogy: choosing the friend’s memory that looks most like the new photo.
- The `PredictionResult` includes confidence numbers (how sure the system is) and a small gating decision (ACT) that can ask for help if uncertain.

!!! note
    The numeric confidence values and `act_action` depend on internal calibration and small example images; exact numbers will vary. [TODO: Verify against learner.py]

**Expected Output & Resource Usage**

- Expected output: a printed `PredictionResult` object, e.g. `PredictionResult(prediction='leaf', raw_confidence=0.1, calibrated_confidence=0.4, neighbor_idx=0, uncertainty_flag=False, act_action='accept')` (values will vary). See `PredictionResult` and `predict` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- Latency: typically small on CPU — a local run with tiny images should complete in about ~100 ms per prediction on modern laptops. This is a rough estimate; real time depends on machine CPU speed and Python interpreter.
- Memory: demo runs with three tiny images should stay well under 200 MB of RAM. AdaptShot is CPU-first and keeps memory usage modest by design.

!!! tip
    If your environment is slower or you see larger memory usage, try the script with smaller images (e.g., 16×16) or close other applications.

**Verification Checklist**

- [ ] I can run the 8-line script (`first_predict.py`) from the repository root.
- [ ] I see a printed `PredictionResult` after the script runs.
- [ ] I can explain what each line in the script does (use the Line-by-Line section above).
- [ ] I know where to find the learner implementation: [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py)

If something fails, start with the import verification step in the Installation section. For behavior questions about `predict()`, `load_support_images()`, `save()`, and `load()`, consult the implementation at [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
