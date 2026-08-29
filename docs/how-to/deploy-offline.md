# Deploy offline

> **For:** someone putting AdaptShot or Tambua on a machine that will not have the network — a field office laptop, an extension officer's tablet, a small server, a Raspberry Pi. Assumes the tutorials.

## The one principle

**Everything that needs the network happens once, on a connected machine; the deployed machine needs nothing.** AdaptShot is built so that this is true by default: the model is inside the package, no weights are fetched at runtime, and the test suite fails if the library opens a connection. What remains is getting the package there.

## Prepare on a connected machine

Download the wheel and its three dependencies for the *target's* Python and architecture:

```bash
mkdir wheels
pip download adaptshot --dest wheels --python-version 3.11 --platform manylinux2014_x86_64 --only-binary=:all:
```

For a Raspberry Pi 4 or 5 (64-bit OS) use `--platform manylinux2014_aarch64`. For a Windows laptop, `--platform win_amd64`. The folder is about 60 MB. Copy it to the target on a USB stick.

If you use Tambua, add `pip download tambua --dest wheels ...` (once it is on PyPI) or copy the repository's `apps/tambua` folder.

## Install on the target, with no network

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links wheels adaptshot
python -c "import adaptshot; print(adaptshot.check_environment(measure=False))"
```

`--no-index` refuses to touch the network; if a wheel is missing, pip says which. The last line confirms *predict* is available on that machine.

## Bring a taught learner, or teach on site

Either copy a saved learner's two files ([save, load and migrate](save-load-and-migrate.md)) or teach from photographs taken on site. **Prefer the second.** The prediction sets' promise holds for photographs that resemble the teaching ones; photographs taken with the deployed camera, in the deployed light, are the ones that count. Ten minutes of photographing on the first day is worth more than any learner brought from elsewhere.

## Scenarios

**One laptop, one person.** The tutorials as they stand. A script, or Tambua in a browser on the same machine (`tambua --config yours`).

**A tablet or phone-class device.** AdaptShot runs wherever Python 3.10 and onnxruntime run, and CI validates it on ARM on every change — the full test suite, the ONNX export, and a device profile on a Neoverse-N2 server core, in the README's device table. A server core is not a phone: measure on the device itself with `python -m benchmarks.run_device` and treat its numbers, not the README's, as the specification. The lowest device actually measured is the one in that table.

**A small server for several users.** Run Tambua with `--host 0.0.0.0 --port 7860` on the office network. One learner is shared, so one person's corrections teach everyone's — decide who is allowed to correct. Do not expose it to the internet; there is no authentication.

**A container.** The image below is offline after build:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY wheels /wheels
RUN pip install --no-index --find-links /wheels adaptshot
# Nothing is fetched at runtime: the backbone and the sample photographs are in the wheel.
CMD ["python", "-c", "import adaptshot; print(adaptshot.check_environment())"]
```

Replace the `CMD` with your script or with `tambua --host 0.0.0.0` after installing it the same way.

## Updating

A new version is a new wheel in the `wheels` folder and `pip install --no-index --find-links wheels --upgrade adaptshot`. Saved learners from the previous version load with a migration warning and can be re-saved. Read the [changelog](../reference/changelog.md) entry first; it says whether the prediction sets' contents changed.

## What to check before you leave the site

```bash
python examples/demo/demo.py --no-color
```

with the network off, if you have the repository; otherwise the [environment check](check-what-this-machine-can-do.md) and one prediction on a photograph taken there. If either fails, you have found a problem while you can still fix it.
