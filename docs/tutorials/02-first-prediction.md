# Your first prediction

> **For:** someone who has finished [installing](01-install.md) and has never run a Python script. Ten minutes. No internet needed. The photographs come with AdaptShot, so nothing here depends on you having images yet.

## What you are about to do

Teach AdaptShot three maize-leaf conditions from eleven photographs, then show it a twelfth it has never seen and read what it says back. Seven lines of code, each explained.

## Step 1 — Make a script file

In the `adaptshot-start` folder from the last page, create a plain text file called `first.py`. Any text editor works — Notepad on Windows, TextEdit on Mac (choose *Format → Make Plain Text* first), or [VS Code](https://code.visualstudio.com/) on any system. Paste this into it and save:

```python
from adaptshot import FewShotLearner
from adaptshot.data import sample_images

paths, labels = sample_images()
learner = FewShotLearner()
learner.load_support_images(paths[:-1], labels[:-1])
result = learner.predict(paths[-1])
print(result.prediction, f"{result.calibrated_confidence:.0%}")
```

## Step 2 — Run it

In the terminal, with `(.venv)` showing at the prompt:

```bash
python3 first.py
```

After a second or two it prints one line, something like:

```text
northern_leaf_blight 66%
```

That is AdaptShot's answer for the twelfth photograph: it thinks the leaf shows northern leaf blight, and it is 66% confident. The twelfth photograph *is* northern leaf blight — so it was right, and honest about not being certain.

## What each line did

```python
from adaptshot import FewShotLearner
from adaptshot.data import sample_images
```
Two imports. `FewShotLearner` is the thing that learns and predicts. `sample_images` hands you the twelve photographs that ship with the package — real maize leaves from the PlantVillage dataset, four each of *healthy*, *gray leaf spot* and *northern leaf blight*.

```python
paths, labels = sample_images()
```
Two lists of twelve. `paths` are file locations; `labels` say what each photograph shows. They line up: `labels[3]` describes `paths[3]`.

```python
learner = FewShotLearner()
```
A new, empty learner with sensible defaults. It does not know anything yet.

```python
learner.load_support_images(paths[:-1], labels[:-1])
```
The teaching step. `paths[:-1]` means "all but the last" — eleven photographs with their labels. AdaptShot looks at each one, turns it into a list of numbers that captures what it looks like (an *embedding*), and averages the numbers per label to make one *prototype* for each condition. It also works out, from those same eleven, how confident it should be and what "unusual" looks like. All of this takes about a second on an ordinary laptop.

```python
result = learner.predict(paths[-1])
```
The question. The twelfth photograph is turned into numbers the same way and compared with the three prototypes. The nearest one wins. `result` holds much more than the winner — the next pages read the rest.

```python
print(result.prediction, f"{result.calibrated_confidence:.0%}")
```
Print the winner and the confidence as a percentage. `:.0%` is Python's way of saying "show as a whole-number percent".

## Step 3 — Look at everything it gave back

Add one line to the end of `first.py` and run it again:

```python
print(result)
```

You will see every field, including some that matter more than the winner:

- `conformal_set` — the short list it stands behind (page 4 explains the promise attached to it)
- `uncertainty_flag` and `act_action` — whether it wants a person to look
- `ood_flag` — whether the photograph looked like nothing it was taught

Do not worry about reading them all yet. [Page 4](04-reading-the-answer.md) goes through each one and says what to do about it.

## Why the photographs are included

Because a first example must not be able to fail. If it needed your own pictures, the first error you hit would be about file paths, not about AdaptShot, and you would have no way to tell which. The twelve leaves are real photographs, unmodified, with their source and licence recorded next to them inside the package (`adaptshot/data/samples/README.md`).

Next: [your own photographs](03-your-own-photos.md).
