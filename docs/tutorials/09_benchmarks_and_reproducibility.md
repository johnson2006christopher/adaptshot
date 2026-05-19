---
title: "09 — Benchmarks and Reproducibility"
nav_order: 9
---

This chapter shows how to measure AdaptShot with the repository's real benchmark scripts. We use only the benchmark tools that exist in `benchmarks/` and the utilities they call.

!!! note
    A benchmark is a repeatable test that helps you understand speed, accuracy, and stability. Think of it like timing the same lap on the same track under controlled conditions.

## 1. The Main Smoke Benchmark

The core benchmark harness is [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py). It runs a CPU-only few-shot evaluation using CIFAR-10, extracts support embeddings, predicts query images, and prints results.

The benchmark uses these real pieces:

- `AdaptShotConfig` from [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py)
- `extract_embedding()` from [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py)
- `find_nearest_neighbor()` from [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py)
- `set_deterministic_seed()` and `verify_determinism()` from [src/adaptshot/utils/determinism.py](src/adaptshot/utils/determinism.py)

### Run It

```bash
python -m benchmarks.run_benchmark --smoke-test --seed 42 --output results/smoke_test.json
# Expected output: a printed summary with accuracy, average latency, p95 latency, embedding time, and a determinism check result
```

### What The Harness Returns

The JSON output includes these real fields:

| Field | Meaning |
|---|---|
| `accuracy` | Fraction of query examples predicted correctly |
| `latency_avg_ms` | Average per-query prediction latency in milliseconds |
| `latency_p95_ms` | 95th percentile per-query latency in milliseconds |
| `embedding_time_s` | Time spent extracting support embeddings |
| `support_size` | Number of support examples used |
| `query_size` | Number of query examples used |
| `config` | The config values used for the run |

Analogy: the JSON file is like a lab notebook page that records exactly how the test was run.

## 2. How The Smoke Benchmark Works

`load_few_shot_split()` in [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py) loads a CIFAR-10 subset and picks support and query examples deterministically.

`run_smoke_test()` then:

1. extracts support embeddings
2. predicts each query image
3. measures latency
4. writes the metrics to JSON
5. verifies determinism by re-running extraction

### Important Constraint

The smoke benchmark is CPU-first and explicitly disables FAISS in its default config. That means the benchmark measures the plain library path, not a GPU or cloud version.

## 3. Energy And Carbon Profiling

If you want energy-focused metrics, use [benchmarks/energy_profile.py](benchmarks/energy_profile.py). That script measures wall time, process time, memory, and estimates Joules and CO₂.

### Run It

```bash
python -m benchmarks.energy_profile --smoke-test --seed 42
# Expected output: JSON with baseline and eco sections, plus joules_estimate, co2_g_estimate, latency_avg_s, latency_p95_s, and deterministic
```

The script uses:

- `set_deterministic_seed()` and `verify_determinism()` from [src/adaptshot/utils/determinism.py](src/adaptshot/utils/determinism.py)
- `extract_embedding()` from [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py)
- `find_nearest_neighbor()` from [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py)

The estimate fields are not a hardware power meter; they are benchmark estimates computed by the script itself.

### Interpreting The Energy Output

| Field | What It Tells You | Analogy |
|---|---|---|
| `latency_avg_s` | Average time per query | How long one shop customer waits |
| `latency_p95_s` | Slow-tail query time | The slowest line at the checkout |
| `joules_estimate` | Estimated energy use | Fuel burned for the trip |
| `co2_g_estimate` | Estimated carbon output | Emissions from that trip |
| `eco_mode` | Whether early-exit was enabled | Taking a shorter route when the answer is obvious |

!!! warning
    These are script-generated estimates. Do not present them as universal truth or copy them without running the benchmark on your own machine.

## 4. Day 2 And Day 3 Integration Scripts

The repository also includes integration simulations that demonstrate continual-learning behavior over multiple steps.

- [benchmarks/day2_integration.py](benchmarks/day2_integration.py) shows calibration and feedback routing over a short stream.
- [benchmarks/day3_integration.py](benchmarks/day3_integration.py) shows ACT, calibration, feedback routing, and CA-EWC together.

These scripts use real classes from `src/adaptshot/`:

- `CalibrationEngine` in [src/adaptshot/core/calibration.py](src/adaptshot/core/calibration.py)
- `FeedbackRouter` and `Correction` in [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py)
- `ACTEngine` in [src/adaptshot/core/act.py](src/adaptshot/core/act.py)
- `CAEWCFinetuner` in [src/adaptshot/training/finetune.py](src/adaptshot/training/finetune.py)

Analogy: these are practice matches, not just single drills. They show how the components behave when they are used together.

## 5. What To Check In Every Benchmark Run

| Check | Why It Matters |
|---|---|
| Same seed | Ensures repeatable splits and outputs |
| CPU-only config | Matches the library's default deployment target |
| Output JSON saved | Lets you compare runs later |
| Determinism pass | Confirms the library is stable under repeated runs |
| No invented metrics | Keeps the report honest |

## 6. Practical Workflow

### Step 1: Run the smoke benchmark

```bash
python -m benchmarks.run_benchmark --smoke-test --seed 42
```

### Step 2: Save the output

```bash
python -m benchmarks.run_benchmark --smoke-test --seed 42 --output results/baseline.json
```

### Step 3: Compare one change at a time

If you change `backbone`, `eco_mode`, `use_faiss`, or `max_buffer_size`, re-run the same benchmark so the result is comparable.

### Step 4: Use the source as truth

If a benchmark result looks surprising, open [benchmarks/run_benchmark.py](benchmarks/run_benchmark.py) or [benchmarks/energy_profile.py](benchmarks/energy_profile.py) and inspect the code path directly.

## 7. Common Benchmark Mistakes

| Mistake | What Happens | Fix |
|---|---|---|
| Changing the seed between runs | Results stop matching | Use the same `--seed` value |
| Comparing different hardware | Numbers are not directly comparable | Benchmark on the same machine when possible |
| Using a different dataset split | Accuracy changes for reasons unrelated to code | Keep the benchmark split fixed |
| Copying example numbers from docs | The numbers may not match your machine | Run the benchmark yourself |

## 8. Verification Checklist

- [ ] I can run `python -m benchmarks.run_benchmark --smoke-test --seed 42`.
- [ ] I can explain what `accuracy`, `latency_avg_ms`, and `latency_p95_ms` mean.
- [ ] I can run `python -m benchmarks.energy_profile --smoke-test --seed 42`.
- [ ] I know which fields in the energy profile are estimates.
- [ ] I can point to the real source files used by the benchmark scripts.
