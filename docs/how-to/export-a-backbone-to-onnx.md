# Export a backbone to ONNX

> **For:** a contributor or an advanced user who wants a backbone to run without PyTorch — on a machine that cannot have it, or to ship it. Needs the torch and onnx extras. Maintainer-level; most people never need this page.

## Why ONNX

The standard install runs inference through onnxruntime, with the backbone's graph and weights bundled in the wheel. That is what makes `pip install adaptshot` 3.5 MB, the process 120 MB, and the whole thing torch-free. A backbone that exists only as a torchvision model needs torch at runtime; exporting it produces a graph onnxruntime can run alone.

## Export

```bash
pip install "adaptshot[torch,onnx]"
python scripts/export_backbones.py --backbone mobilenet_v3_small --output-dir /tmp/onnx-out --verify
```

`--verify` embeds a fixed input through both the torch model and the exported graph and reports the agreement. `--all` exports every registered backbone. Each export produces `<name>.onnx` (the graph), `<name>.onnx.data` (the weights, stored externally), and `<name>.json` (opset, input shape, SHA-256, size).

**Write to a scratch directory, not into `src/adaptshot/data/`.** The script's default output directory is the package's own data folder, and anything placed there is picked up at runtime by name — which is fine on your machine and misleading if you then build a wheel: a stale `build/` or `egg-info` from an earlier build can package whatever was on disk. `CLAUDE.md` records the 44 MB wheel that taught us this.

## What ships, and why only one

| backbone | graph | weights | total |
|---|---|---|---|
| `mobilenet_v3_small` | 0.31 MB | 3.68 MB | **4.0 MB** — bundled |
| `resnet18` | 0.09 MB | 44.7 MB | 44.8 MB — not bundled |

The wheel's contents are named explicitly in `pyproject.toml` (`package-data`), not globbed, so what ships is a decision rather than whatever happened to be on disk. Adding a backbone to the wheel means adding its two files there and updating `bundled_onnx_backbones()`'s expectations in the tests.

## Agreement with torch

`tests/test_onnx_parity.py` enforces that the bundled graph and the torch path agree to within 1e-4 absolute (measured: 1.8e-6) and to cosine > 0.9999 on every bundled backbone. A re-export at a different opset or from different pretrained weights would move every downstream number silently; the test is what makes it loud. `python -m benchmarks.onnx_parity` reports the same agreement alongside latency, each in its own process, because the two runtimes' thread pools contend if measured together.

## Using an exported graph without bundling it

Place `<name>.onnx` and `<name>.onnx.data` in `adaptshot/data/` inside the installed package (find it with `python -c "import adaptshot.data, os; print(os.path.dirname(adaptshot.data.__file__))"`). `bundled_onnx_backbones()` reads that directory at runtime, so the backbone becomes available by name on that install without torch. This is a local convenience, not a distribution mechanism.
