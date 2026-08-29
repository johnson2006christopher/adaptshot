# Run the demo with the wifi switched off

> **For:** someone presenting AdaptShot to others — a conference, a classroom, a farmer group — who needs it to work with no network at all. Assumes AdaptShot is installed and you have a copy of the repository.

## What the demo shows

In under two seconds, on a laptop CPU: a learner taught from twelve maize-leaf photographs; four leaves it has never seen, including one from a crop it was never taught; the prediction set widening as it becomes less sure; and the moment it declines and asks for a person. Then a real measured coverage figure from the published benchmark, read from the results file, not typed into the script.

## Run it

From the repository root:

```bash
python examples/demo/demo.py
```

Options:

| flag | effect |
|---|---|
| `--pause` | wait for Enter between steps — for presenting |
| `--no-color` | plain text, for a terminal or projector that mangles colour |
| `--debug` | show a full traceback if something goes wrong, instead of one line |

## Why it cannot secretly use the network

The script disables outbound connections in its own process before importing AdaptShot. If anything in the chain tried to download a model or phone home, it would fail loudly — on your machine, before the talk — rather than succeed silently on the venue wifi and fail later without it. The project's CI runs the same demo inside a network namespace with no interfaces at all on every change.

If you want the same certainty on your own machine, on Linux:

```bash
unshare -rn python examples/demo/demo.py
```

runs it in a namespace with no network. (Some Ubuntu versions restrict this; `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` lifts the restriction for the session.)

## Rehearse it once

Run it in the room, on the laptop, with the wifi off, before the audience arrives. Two consecutive runs print byte-identical output apart from timings, so what you rehearse is what they will see. The demo prints *"coverage artifact not found"* and how to produce it if `results/plantvillage_5way5shot.json` is missing — it still finishes, so the talk does not stop; but you want that file present.

## The one-page handout

`examples/demo/HANDOUT.md` has the install command, the seven-line quickstart, the measured numbers including the unflattering one, and the repository URL. Print it; add a QR code of the URL. People decide to try something in the ten seconds after a talk ends.

## Verify it works before you leave

```python
import subprocess, sys
completed = subprocess.run([sys.executable, "examples/demo/demo.py", "--no-color"], capture_output=True, text=True)  # docs: not run -- needs the repository checkout, not just the package
print(completed.returncode == 0 and "ask a human" in completed.stdout)
```

That is what `tests/test_demo.py` checks on every push: exit code zero, the abstention shown, the coverage figure displayed, under two minutes.
