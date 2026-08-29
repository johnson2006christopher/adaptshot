# Run the benchmarks and reproduce every published number

> **For:** someone who wants to check a figure in the README or the technical note on their own machine, or measure a change they made. Assumes a repository checkout with the `dev` extra installed. The full runs need one manual download.

## The rule behind all of them

No number appears in the documentation that a test does not trace back to a committed file in `results/`. Each benchmark below writes one of those files, records the hardware it ran on, and is reproducible from a seed. If you re-run one and get a different figure, that is a finding — file it with the artifact.

## The smoke benchmark — runs offline, in CI, on every change

```bash
python -m benchmarks.run_benchmark --smoke-test --seed 42
```

Uses CIFAR-10 if it is already cached under `data/`, otherwise a synthetic fixture, and **never downloads without `--allow-download`**. Reports accuracy (only on real data), latency, and whether two runs at the same seed agree — the determinism check that CLAUDE.md requires of every change. `--backbone mobilenet_v3_small` measures the bundled backbone; the default is `resnet18` for comparability with the project's history, and needs the torch extra. Writes `results/smoke_test.json` (default backbone) or `results/smoke_test.<backbone>.json`.

## The PlantVillage benchmark — the published result

One manual download, pinned to a content-addressed commit and verified against a SHA-256 manifest:

```bash
python scripts/fetch_plantvillage.py --out data/pv_bench --per-class 20 --preset benchmark
python scripts/fetch_plantvillage.py --out data/pv_bench --verify
```

Then:

```bash
python -m benchmarks.run_plantvillage --seed 42
```

5-way 5-shot, 100 episodes, every method on the same episodes and embeddings: AdaptShot against nearest-centroid, k-NN and a logistic-regression probe; conformal coverage and set size against a calibrated top-1 threshold; latency by stage as median and p95; cold start; peak memory for one cycle *and* for the harness, named apart. Writes `results/plantvillage_5way5shot.json`. Two to three minutes on a laptop.

Run it on the standard install, not one with torch: the memory figure is meant to describe the library, and the technical note's numbers came from a core-install run.

## The shift benchmark — where the guarantee bends

```bash
python -m benchmarks.run_shift --seed 42
```

Same pool, queries blurred, darkened, re-compressed and downscaled while the support is left alone; coverage and set size per cell; the OOD flag as an early-warning signal; and the recovery after ten in-situ corrections through `correct()`. Writes `results/plantvillage_shift.json`. About fifty minutes — it is sixteen cells of forty episodes, twice.

## The ONNX parity check

```bash
python -m benchmarks.onnx_parity
```

Embeds the same image through the bundled ONNX backbone and through torch, in separate processes, and reports agreement (cosine and max absolute difference) and latency as min and median across processes. Needs the torch extra. The agreement is also enforced by `tests/test_onnx_parity.py`.

## The synthetic guarantee harness

Not a benchmark — a test — but it is where the conformal guarantee is validated statistically, and it runs in seconds:

```bash
pytest tests/test_conformal_coverage.py -q
```

Five overlapping Gaussian classes; α ∈ {0.01, 0.05, 0.1, 0.2} × calibration sizes {10, 20, 50, 200}; coverage asserted against the target with a tolerance derived from the trial-level standard error; set size asserted in both directions. The [guarantee explanation](../understand/the-guarantee.md) says what each cell means.

## Comparing a change

Run the relevant benchmark before and after on the same machine, in the same session, and compare the artifacts. Median latencies on this project's laptop have been stable to within a few percent; p95s have not — the [results reference](../reference/results-and-artifacts.md) records the measured spread so you know what is noise.
