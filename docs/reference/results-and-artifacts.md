# Results and artifacts

> **For:** anyone checking a number — where it came from, what machine produced it, and how to regenerate it. Every figure the documentation quotes traces to one of the files below; `tests/test_docs_claims.py` fails if a document and its artifact disagree.

## The rule

A number appears in the README, the technical note or this documentation only if a committed file in `results/` records it, the command that produced it, the seed, and the hardware. The tests format the figures from the file and assert the documents quote them verbatim. When you re-run a benchmark and get a different figure, that is a finding: file it with the artifact.

## `results/plantvillage_5way5shot.json` — the published result

Produced by `python -m benchmarks.run_plantvillage --seed 42` on the core install ([how to run it](../how-to/run-the-benchmarks.md)).

| key | contents |
|---|---|
| `accuracy.<method>` | mean accuracy over episodes and 95% half-width, for `adaptshot`, `nearest_centroid`, `knn_1`, `knn_5`, `linear_probe` — all on the same episodes and embeddings |
| `conformal` | `alpha`, `target_coverage`, `empirical_coverage`, `mean_set_size`, `ood_flag_rate` (each mean ± half-width) |
| `top1_threshold` | the alternative: top-1 with a threshold calibrated per episode to the same target — `accuracy`, `coverage`, `mean_set_size`, `threshold` |
| `timing` | `embedding_ms`, `support_fit_ms`, `predict_ms` as median / p95 / n; `cold_start` as seconds and that fresh process's `peak_rss_mb`; `benchmark_process_peak_rss_mb` for the harness itself; a `note` saying which is which |
| `protocol` | task, episodes, calibration and query sizes per class, seed, backbone, α |
| `dataset` | PlantVillage repository, pinned commit, licence, citation, preset, file count |
| `hardware` | CPU model, core count, RAM, OS, Python, numpy, onnxruntime, library version, and which install (`core` or `torch`) ran it |

Two memory numbers are deliberately named apart: `cold_start.peak_rss_mb` describes one process teaching eleven photographs and answering one query — the library; `benchmark_process_peak_rss_mb` describes the harness holding 400 embeddings and four baselines. Only the first is a claim about AdaptShot.

## `results/plantvillage_shift.json` — coverage under distribution shift

Produced by `python -m benchmarks.run_shift --seed 42` on the core install.

| key | contents |
|---|---|
| `cells[]` | one per (shift kind, level): `accuracy`, `coverage`, `set_size`, `ood_rate` before, and `after_in_situ_corrections` with `k` — each mean ± half-width; `identity` marks the clean level |
| `early_warning` | Pearson correlation across shifted cells between coverage shortfall and OOD flag rate |
| `protocol` | the suite (kinds and levels), what was shifted, the mitigation, `recalibrate_k`, α, episodes, seed |
| `dataset`, `hardware` | as above |

## `results/smoke_test.json` — the CI smoke benchmark

Produced by `python -m benchmarks.run_benchmark --smoke-test --seed 42`. The committed file records **68% accuracy on the 5-way 10-shot CIFAR-10 smoke split** with `resnet18` — a smoke check, not a result, and quoted here only so a test can hold this page to the file. `accuracy` is present only when CIFAR-10 was cached (`data_source: cifar10`); on the synthetic fixture it is `null` and only latency and determinism are reported. `results/smoke_test.<backbone>.json` holds runs on a non-default backbone so the canonical file is never overwritten by one.

## `results/device_<machine>.json` — what one machine costs

Produced by `python -m benchmarks.run_device --seed 42` on that machine, on a core install. Two are committed: `device_x86_64.json` from the laptop the other artifacts came from, and `device_aarch64.json` from the ARM runner CI uses (the `device-arm` job regenerates it on every change and uploads it as a workflow artifact; the committed copy is the one the README quotes). Fields:

- `hardware` — CPU model (from `/proc/cpuinfo`, or `lscpu` on ARM), core count, RAM, platform, Python, numpy, and `install: core`. A profile taken with torch installed is rejected by the test.
- `quickstart` — the README's split, eleven teach and one asks, with the predicted and expected label and `correct`.
- `timing.embedding_ms`, `timing.predict_ms` — median, p95 and n over the bundled photographs, after warm-up; `timing.support_fit_ms` — one `load_support_images` of eleven.
- `cold_start` — fresh interpreter from before `import adaptshot` to the first answer, repeated: `seconds_min`, `seconds_median`, `peak_rss_mb_max`, `n`.
- `export` — a graph exported on that machine compared with the bundled one through onnxruntime: `verified`, cosine and max absolute difference, file sizes, providers.

## The measured spread, so you know what is noise

On the laptop these were taken on (11th Gen Intel i7-11800H, 16 cores, 31 GB, Linux):

- **Median latencies** were stable within a few percent across runs.
- **p95 latencies were not**: the support-fit p95 ranged from 1.5 s to 4.0 s across three runs while its median held within 2%. The slow episodes fall mid-run, not at the start. The tail is the machine's; plan around the median.
- **Cold start** varied between 0.9 s and 1.3 s across runs.
- **resnet18 latency is bimodal** on this machine — about 6.7 ms or about 11 ms per process, decided at start-up. `benchmarks/onnx_parity.py` reports min and median across processes so the spread is visible rather than averaged away.
- **This laptop is faster than the project's target hardware.** The figures are what the library costs *here*. The README's device table puts the same measurement on an ARM server core beside it (#31); phone-class hardware is still unmeasured.

## Regenerating everything

```bash
python scripts/fetch_plantvillage.py --out data/pv_bench --per-class 20 --preset benchmark   # once, network
python -m benchmarks.run_benchmark --smoke-test --seed 42
python -m benchmarks.run_plantvillage --seed 42
python -m benchmarks.run_shift --seed 42
pytest tests/test_docs_claims.py -q   # the documents still match
```

Run the two PlantVillage benchmarks one at a time on an otherwise idle machine; the latency fields are contaminated by anything else using the CPU.
