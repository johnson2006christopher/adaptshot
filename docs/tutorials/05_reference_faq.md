---
title: "05 — Reference & FAQ"
nav_order: 5
---

This reference assumes you've completed tutorials 01–04. Use this page as a compact glossary, API quick reference, error troubleshooting table, and next-steps guide.

## 1. Plain-English Glossary (20 terms)

1. few-shot — A learning setting where the system learns from a very small number of examples (e.g., 1–10 photos). Analogy: teaching a friend with a handful of photos instead of a whole album. See `FewShotLearner` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
2. support set — The small set of labeled examples you provide to teach the model; passed to `load_support_images()`. Analogy: the sticky-note labeled photos you show a friend. See `load_support_images()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
3. embedding — A numeric fingerprint (vector) representing an image's visual features. Analogy: a short checklist describing a photo. Produced by `extract_embedding()` in [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py).
4. preview_signature — A cheap, truncated summary of an embedding used for fast early-exit checks. See `compute_preview_signature()` in [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py).
5. nearest neighbor — The stored support example whose embedding is closest to a query embedding; used for prediction. See `find_nearest_neighbor()` in [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py).
6. PredictionResult — Structured return from `predict()` containing `prediction`, `raw_confidence`, `calibrated_confidence`, `neighbor_idx`, `uncertainty_flag`, and `act_action`. See dataclass in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
7. calibration — The process of converting raw similarity scores into probability-like scores that better match real-world correctness. Analogy: a scale that learns to weigh accurately over time. See `CalibrationEngine` in [src/adaptshot/core/calibration.py](src/adaptshot/core/calibration.py).
8. calibrated_confidence — The confidence after calibration, produced during `predict()`; found in `PredictionResult`. See `predict()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
9. raw_confidence — The initial similarity-based score before calibration; returned by the similarity search. See `find_nearest_neighbor()` in [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py).
10. ACT (Adaptive Computation Time) — A gating mechanism that decides whether to accept a prediction or ask for human help (`act_action`). See `ACTEngine` in [src/adaptshot/core/act.py](src/adaptshot/core/act.py).
11. act_action — Short string returned by ACT (e.g., `accept` or `query`) indicating the action to take. Produced in `predict()`; see [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
12. CA-EWC (Continual Adaptation with Elastic Weight Consolidation) — A light fine-tuning strategy used by `CAEWCFinetuner` to update a small model head without catastrophic forgetting. See [src/adaptshot/training/finetune.py](src/adaptshot/training/finetune.py).
13. UP-UGF (Uncertainty + Recency + Redundancy pruning) — Buffer pruning strategy implemented by `UPUGFPruner` to keep replay buffer size bounded. See [src/adaptshot/training/up_ugf.py](src/adaptshot/training/up_ugf.py).
14. eco_mode — A boolean config flag that enables fast preview checks and early-exit to save compute; set in `AdaptShotConfig`. See [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py).
15. early_exit_threshold — Float threshold controlling eco-mode early exit behavior (0.5–1.0). See `AdaptShotConfig` in [src/adaptshot/config/settings.py](src/adaptshot/config/settings.py).
16. finetuner — The optional `CAEWCFinetuner` attached to the learner for lightweight fine-tuning of the model head. See `FewShotLearner._init_or_rebuild_model_head()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
17. FeedbackRouter — Component that routes human corrections into buffer and possibly triggers fine-tuning. See [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py).
18. integrity checksum — SHA-256 checksums stored alongside checkpoints to detect corruption; built in `_build_integrity_payload()` and checked in `_load_state_payload()` in [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
19. deterministic smoke test — A reproducible profiling run provided by `benchmarks/energy_profile.py` that estimates latency, joules, and CO₂. See [benchmarks/energy_profile.py](benchmarks/energy_profile.py).
20. support_embedding_cache — An internal cache of a support embedding + preview used to speed repeated queries; set via `set_support_embedding_cache()` in [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py).

## 2. Quick API Reference Table

| Method | Inputs | Output | Source |
|---|---:|---|---|
| `FewShotLearner()` | `config: Optional[AdaptShotConfig]` | `FewShotLearner` instance | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| `load_support_images(image_paths, labels)` | `List[str], List[str|int]` | `None` (initializes support set) | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| `predict(image)` | `image: str|PIL.Image|np.ndarray` | `PredictionResult` | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| `correct(image_path, true_label, confidence_weight=1.0)` | `str, str|int, float` | `Dict[str, Any]` routing summary | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) and [src/adaptshot/training/feedback_router.py](src/adaptshot/training/feedback_router.py) |
| `save(path)` | `str` | `None` (writes JSON + .npy + optional head.pt)` | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| `load(path)` | `str` | `FewShotLearner` (restored) | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| `extract_embedding(image, config)` | `PIL.Image, AdaptShotConfig` | `np.ndarray` embedding | [src/adaptshot/core/extractor.py](src/adaptshot/core/extractor.py) |
| `find_nearest_neighbor(query, support_embeddings, support_labels, use_faiss=False)` | `ndarray, ndarray, ndarray, bool` | `(label, raw_confidence, neighbor_idx)` | [src/adaptshot/core/similarity.py](src/adaptshot/core/similarity.py) |

## 3. Common Errors & Fixes

| Error message (exact) | Why it happens | How to fix | Source |
|---|---|---|---|
| "Support set cannot be empty. Provide at least one RGB image path and label. See docs/getting-started/quickstart.md." | `load_support_images()` received empty lists | Provide at least one valid RGB image path and matching label; ensure upload pipeline stores files first | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| "Image file not found: '{image_path}'. Verify the path and try again." | File path is missing or incorrect | Check file path, permissions, and working directory; use absolute paths in services | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| "Expected 3-channel RGB image, got 1-channel grayscale array. Convert before loading. See docs/getting-started/quickstart.md." | Input image is grayscale or has wrong channels | Convert to RGB (e.g., Pillow `img.convert('RGB')`) before calling `predict()` | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| "Calibration window is not ready. Need at least {min_samples} observations, got {observed}. Continue collecting feedback with correct()." | Calibration requires more correction examples for temperature method | Continue collecting corrections via `correct()`; accept raw confidence until calibration ready | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| "State file not found: '{path}'. Ensure the path is correct before loading." | `load()` path incorrect or file missing | Verify checkpoint paths and storage mounts; use atomic save pattern to avoid partial writes | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |
| "Checkpoint integrity check failed. The checkpoint may be corrupted or tampered with." | Checksum mismatch between stored metadata and embeddings | Restore a known-good checkpoint or re-create via `save()`; ensure storage is reliable | [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py) |

## 4. When to Use AdaptShot vs. Other Tools

Note: keep comparisons factual and constraint-driven.

| Task / Constraint | Use AdaptShot when... | Use alternatives when... |
|---|---|---|
| Few-shot image classification on CPU-limited devices | You need model behavior from a handful of labeled images, CPU-only, with lightweight local inference and optional human feedback | Skip AdaptShot and use larger pretrained pipelines if you require full fine-tuning on GPUs or large datasets |
| Large-scale training or transfer learning | Not ideal — AdaptShot is optimized for small support sets and lightweight head updates | Use Hugging Face Transformers or PyTorch training loops for large-scale fine-tuning with GPUs |
| Object detection in images/video (bounding boxes) | Not suitable — AdaptShot focuses on image-level few-shot classification | Use YOLO-family models or Detectron2 for detection tasks with annotated boxes |
| Standard classical ML on tabular data | Not applicable — AdaptShot is image-focused | Use scikit-learn for robust, well-tested tabular algorithms (SVMs, RandomForest) |

## 5. Next Steps & Community

- Contribute: follow repository [CONTRIBUTING.md](CONTRIBUTING.md) and open PRs against the `main` branch.
- Report bugs or ask questions: open GitHub Issues in the repository and include reproducer scripts and exact error messages from logs.
- Roadmap & design notes: see [ROADMAP.md](ROADMAP.md) and [AGENTS.md](AGENTS.md) for project-level context.
- Local development commands: lint with `./venv/bin/ruff check src/ tests/ --fix`, typecheck with `./venv/bin/mypy src/adaptshot --strict`, run targeted tests with `./venv/bin/pytest tests/test_persistence.py -v`.

## 6. Final Verification Checklist

- [ ] I can find definitions for 20 key terms in this glossary and link them to code.
- [ ] I can call `load_support_images()`, `predict()`, and `correct()` and map outputs to `PredictionResult` fields (`prediction`, `calibrated_confidence`, `uncertainty_flag`, `act_action`). See [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
- [ ] I know where to run the deterministic energy smoke test (`benchmarks/energy_profile.py`) and read `joules_estimate` and `co2_g_estimate`.
- [ ] I can handle the common errors in section 3 and trace them to the listed source files.
- [ ] I can contribute via `CONTRIBUTING.md` and open issues with reproducible examples.

---

*Created by [Johnson Christopher Hassan](https://github.com/johnson2006christopher)*  
*Connect on [LinkedIn](https://www.linkedin.com/in/johnson-hassan-935124311/)*  
*Project: [github.com/johnson2006christopher/adaptshot](https://github.com/johnson2006christopher/adaptshot)*
