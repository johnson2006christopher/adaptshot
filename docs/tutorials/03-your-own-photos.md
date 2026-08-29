# Your own photographs

> **For:** someone who has run the [first prediction](02-first-prediction.md) and wants AdaptShot to recognise their own things. Twenty minutes. Offline.

## The one rule

**One folder per thing you want recognised, with that thing's name as the folder name.** Inside each folder, photographs of it. That is the whole data format. There is no spreadsheet, no labelling tool, no special file type — a folder called `healthy` with pictures of healthy leaves in it.

```text
my-leaves/
  healthy/
    IMG_0412.jpg
    IMG_0413.jpg
    ...
  gray_leaf_spot/
    IMG_0501.jpg
    ...
  northern_leaf_blight/
    ...
```

## Step 1 — Build a folder like that

To make this page runnable without your photographs, the script below builds the folder from the twelve that ship with AdaptShot. When you have your own, skip this step and point the next one at your folder instead.

```python
import shutil
from pathlib import Path
from adaptshot.data import sample_images

root = Path("my-leaves")
for path, label in zip(*sample_images(), strict=True):
    (root / label).mkdir(parents=True, exist_ok=True)
    shutil.copy(path, root / label / Path(path).name)

print(sorted(folder.name for folder in root.iterdir()))
```

Save it as `make_folder.py`, run it, and you have `my-leaves/` with three subfolders:

```text
['gray_leaf_spot', 'healthy_maize', 'northern_leaf_blight']
```

## Step 2 — Teach from the folder, hold one back to test

Save this as `own.py` and run it:

```python
from pathlib import Path
from adaptshot import FewShotLearner

root = Path("my-leaves")
paths, labels = [], []
for folder in sorted(root.iterdir()):
    for image in sorted(folder.glob("*.jpg")):
        paths.append(str(image))
        labels.append(folder.name)

# Keep the last photograph of each class back, so we can check the answers.
test = {label: paths[i] for i, label in enumerate(labels)}   # last path per label wins
teach_paths = [p for p in paths if p not in test.values()]
teach_labels = [l for p, l in zip(paths, labels, strict=True) if p not in test.values()]

learner = FewShotLearner()
learner.load_support_images(teach_paths, teach_labels)

for truth, path in test.items():
    result = learner.predict(path)
    mark = "✓" if result.prediction == truth else "✗"
    print(f"{mark} {Path(path).name:<28} said {result.prediction:<22} ({result.calibrated_confidence:.0%})")
```

The loop at the top walks the folders and builds the same two lists `sample_images()` handed you last time — file paths and the folder name each came from. Then one photograph per class is held back and asked about after teaching. You will see three lines, each a tick or a cross.

Expect a cross now and then. With three teaching photographs per class, a wrong guess is normal; it is what the confidence and the set on the next page are for.

## Step 3 — Point it at your real folder

Change one line:

```python
root = Path("my-leaves")   # docs: not run -- replace with your own folder, e.g. Path("/home/me/photos/leaves")
```

and run again. Nothing else changes.

## What makes a good set of teaching photographs

- **Take them where you will use them.** Same camera, same kind of light, same distance. The promise on the next page holds only when the photographs you ask about resemble the ones you taught with; a set taught from a website and asked about in a field is the way it breaks.
- **Five per class is a floor, not a target.** More helps, up to a point; twenty is plenty. Fewer than three per class and AdaptShot cannot judge what "unusual" looks like for that class, and will tell you so.
- **Vary what does not matter, keep what does.** Different leaves, angles, backgrounds — but all clearly showing the condition the folder is named for.
- **Label carefully.** A misfiled photograph teaches the wrong thing. If you are unsure what a photograph shows, leave it out.
- **Any format Pillow can read** works: JPEG, PNG, WebP, BMP. Colour photographs; AdaptShot converts to RGB.

## Where the folder can live

Anywhere. AdaptShot reads the files once, at `load_support_images`, and keeps only the numbers it computed from them. Moving the folder afterwards does not affect a learner you have already taught — and [page 5](05-teaching-corrections.md) shows how to save that learner so you do not have to teach it again.

Next: [reading the answer](04-reading-the-answer.md) — every field in `result`, and what to do with each.
