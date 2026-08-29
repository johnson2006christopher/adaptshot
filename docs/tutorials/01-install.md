# Install AdaptShot, and find out what your computer can do

> **For:** someone who has never used a terminal or installed a Python package. Fifteen minutes. You need an internet connection for this page only; everything after it works offline.

## What you are about to do

You will open a *terminal* — a window where you type commands instead of clicking — check that Python is installed, make a private folder for AdaptShot to live in, install it, and ask it what your computer is capable of. Every command is shown exactly as you should type it. Copy and paste is fine.

## Step 1 — Open a terminal

- **Windows:** press the Windows key, type `PowerShell`, press Enter.
- **Mac:** press ⌘-Space, type `Terminal`, press Enter.
- **Linux:** it is usually called *Terminal* in the application menu.

A window with a blinking cursor appears. Commands go there.

## Step 2 — Check Python

AdaptShot needs Python 3.10 or newer. Type this and press Enter:

```bash
python3 --version
```

You should see something like `Python 3.12.3`. If the number after the first dot is 10 or higher, you are fine. If you see an error, or a version below 3.10, install Python from [python.org/downloads](https://www.python.org/downloads/) — choose the latest version, and on Windows tick **"Add python.exe to PATH"** during installation — then close the terminal, open a new one, and try again.

On Windows the command may be `python` rather than `python3`. If one does not work, try the other.

## Step 3 — Make a folder and a private Python inside it

A *virtual environment* is a private copy of Python for one project, so that installing AdaptShot cannot interfere with anything else on your computer. Make a folder, go into it, and create one:

```bash
mkdir adaptshot-start
cd adaptshot-start
python3 -m venv .venv
```

Now switch the terminal to use it. This is the one step that differs by system:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **Mac and Linux:** `source .venv/bin/activate`

The prompt changes to show `(.venv)` at the start. That means the private Python is active. You will need to run that activation line again every time you open a new terminal for this folder.

## Step 4 — Install

```bash
pip install adaptshot
```

This downloads AdaptShot and the three libraries it needs — numpy, Pillow and onnxruntime — about 200 MB in total. It takes between ten seconds and a few minutes depending on your connection. When the prompt comes back without red text, it is installed. The model it uses to look at pictures is inside the download; nothing else is fetched later.

## Step 5 — Ask your computer what it can do

Type `python3` and press Enter. The prompt changes to `>>>`: you are now talking to Python directly. Type these two lines, pressing Enter after each:

```python
import adaptshot
print(adaptshot.check_environment())
```

AdaptShot inspects the machine it is running on and prints a report. Every number in it was measured on your computer just now — none is copied from a document. It looks like this (yours will differ):

```text
AdaptShot 0.3.0 -- environment report (everything below was measured here)
  Python 3.12.3 · Linux 6.8 · x86_64 · 4 cores · 5.1 of 7.7 GB RAM free
  numpy 2.2.6   Pillow 11.1.0   torch: not installed   onnxruntime 1.21.0   ...
  bundled backbones: mobilenet_v3_small
  Available now:
    ✓ predict, correct, save / load        41.2 ms per image, median of 5, measured here on mobilenet_v3_small
    ✓ conformal prediction sets            ...
    ✓ out-of-distribution flag             ...
  Not available:
    ✗ fine-tuning (CA-EWC) via correct()   needs torch; download size not measured here (requires the network)
                                           needs: pip install "adaptshot[torch]"
  Fits the 250 MB target here: yes -- this process peaked at 118 MB
```

Three things to read off it:

- **"Available now"** is everything the next four pages use. If `predict` is listed there, you are ready.
- **"Not available"** lists optional extras, each with the exact command to install it. You do not need any of them yet. Fine-tuning needs a large extra library called PyTorch; the [how-to on fine-tuning](../how-to/fine-tune-with-corrections.md) covers it when you want it.
- **The last line** tells you whether AdaptShot fits its memory budget on this machine.

Type `exit()` and press Enter to leave Python.

## If something went wrong

| What you see | What it means | What to do |
|---|---|---|
| `pip: command not found` | The private Python is not active | Run the activation line from Step 3 again |
| `No module named adaptshot` | Same, or the install did not finish | Activate, then run `pip install adaptshot` again and read the last lines |
| `ERROR: Could not find a version that satisfies…` | Python is older than 3.10 | Install a newer Python (Step 2) |
| Red text mentioning `onnxruntime` | Your platform has no prebuilt onnxruntime | Tell us in an [issue](https://github.com/johnson2006christopher/adaptshot/issues) with the full text — this is rare and we want to know |

## What you have now

A folder `adaptshot-start` with a private Python that has AdaptShot installed, and a report of what this machine can do. Next: [your first prediction](02-first-prediction.md), on photographs that came with the download, so nothing depends on you having images yet.
