# Check what this machine can do

> **For:** someone deciding whether a particular computer — a field laptop, a shared server, a Raspberry Pi — can run AdaptShot, and what it would cost there. Ten seconds to run; works offline.

## Ask it

```python
import adaptshot
print(adaptshot.check_environment())
```

Every figure in the report is **measured on the machine running the check**: the latency is a real inference on the bundled photographs, the memory is this process's own high-water mark. Nothing is quoted from a document or from someone else's benchmark.

## Reading the report

```text
AdaptShot 0.3.0 -- environment report (everything below was measured here)
  Python 3.11.9 · Linux 6.1 · aarch64 · 4 cores · 1.9 of 3.8 GB RAM free
  numpy 2.2.6   Pillow 11.1.0   torch: not installed   onnxruntime 1.21.0   faiss: not installed   gradio: not installed
  bundled backbones: mobilenet_v3_small
  Available now:
    ✓ predict, correct, save / load        88.4 ms per image, median of 5, measured here on mobilenet_v3_small
    ✓ conformal prediction sets            coverage guarantee validated in tests/test_conformal_coverage.py; needs ceil((1-alpha)/alpha) calibration scores to be informative
    ✓ out-of-distribution flag             leave-one-out-calibrated Mahalanobis; at least 3 support photos per class
  Not available:
    ✗ fine-tuning (CA-EWC) via correct()   needs torch; download size not measured here (requires the network)
                                           needs: pip install "adaptshot[torch]"
    ✗ backbones other than the bundled one only the bundled backbone(s) without torch
                                           needs: pip install "adaptshot[torch]"
    ✗ faster search for support sets over 100 images   numpy search is used; fine below ~100 images
                                           needs: pip install "adaptshot[faiss]"
  Fits the 250 MB target here: yes -- this process peaked at 118 MB
```

- **Available now** is what works with what is installed. If `predict` is there, the tutorials work.
- **Not available** lists each optional capability with the exact `pip` command that enables it. The download size is deliberately *not* stated: it cannot be measured without the network, and a number copied from elsewhere is what this report exists to avoid.
- **The latency line** is this machine, now. A busy machine reports a slower number — that is the point. Compare with the [published figures](../reference/results-and-artifacts.md) for the machine they were measured on.
- **The memory line** is the process running the check. If you ran this after importing PyTorch, the figure includes PyTorch and the report says so; the 250 MB target describes the standard install.
- **A GPU**, if present, is named and *not selected*. AdaptShot's defaults stay on the CPU on purpose — the [explanation](../understand/why-cpu-only-and-offline.md) says why.

## When you cannot afford the second it takes

```python
import adaptshot
report = adaptshot.check_environment(measure=False)
print([c.name for c in report.capabilities if c.available])
```

`measure=False` skips the inference and the memory reading and returns in under a millisecond — for a start-up check in an application, where you want to know *whether* prediction works, not how fast.

## Using it from code

The return value is an `EnvironmentReport` dataclass, so an application can branch on it:

```python
import adaptshot
report = adaptshot.check_environment(measure=False)
needs = {c.name: c.install for c in report.capabilities if not c.available}
if "fine-tuning (CA-EWC) via correct()" in needs:
    print("Fine-tuning is off on this machine. To enable it:", needs["fine-tuning (CA-EWC) via correct()"])
```

`check_environment` is in the experimental tier of the API: the report's shape may change in a minor release as people say what they need in it. See [API stability](../contributing/api-stability.md).
