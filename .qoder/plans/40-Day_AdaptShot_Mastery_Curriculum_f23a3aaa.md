# 40-Day AdaptShot Mastery Curriculum

## Overview

Each day is designed for 4-6 hours of focused work. The curriculum builds layer by layer:
- **Week 1 (Days 1-5)**: Strong Foundations — data structures, linear algebra, vector math, image processing
- **Week 2 (Days 6-10)**: Core ML Engineering — embeddings, similarity search, calibration, ACT
- **Week 3 (Days 11-15)**: Advanced Algorithms — prototype computation, InfoNCE, k-NN, caching
- **Week 4 (Days 16-20)**: Uncertainty & Reliability — multi-signal uncertainty, OOD, conformal prediction
- **Week 5 (Days 21-25)**: Human-in-the-Loop — feedback routing, EWC, buffer management, explainability
- **Week 6 (Days 26-30)**: Systems Engineering — architecture, API design, testing, profiling, packaging
- **Week 7 (Days 31-35)**: Research Integration — reading papers, implementing from scratch, comparing methods
- **Week 8 (Days 36-40)**: Capstone — redesign AdaptShot subsystems independently, defend every decision

---

## Week 1: Strong Foundations

### Day 1 — The Architecture You Built

**Learning Objectives**: Understand every module in AdaptShot, trace the end-to-end pipeline, read the public API.

**Theory**: Software architecture patterns — Separation of Concerns, Single Source of Truth, Plugin-Ready Design, Composability. Why `FewShotLearner` orchestrates while subsystems implement. The difference between a "library" and a "framework". The frozen dataclass pattern and why immutability matters for ML config.

**Mathematics**: None today — orientation day.

**Algorithms**: None today — orientation day.

**Software Engineering**: 
- Package layout conventions (`src/` layout, `__init__.py` as public API surface)
- Semantic versioning (semver 2.0.0) and why it matters
- The difference between public API (stable) and internal API (unstable)
- Lazy imports: `_get_torch()`, `_get_tv_models()` pattern

**ML Concepts**: What is few-shot learning? The setup: support set, query set, n-way k-shot. Why AdaptShot uses frozen backbones instead of training from scratch.

**AdaptShot Deep Dive**: Module map tour — read `src/adaptshot/__init__.py`, `.openproject.md`, `AGENTS.md`, `pyproject.toml`. Trace how `AdaptShotConfig` flows through every subsystem.

**Code Reading**: 
- `src/adaptshot/__init__.py` (full 44 lines)
- `src/adaptshot/config/settings.py` (full 102 lines)
- `.openproject.md` (the constitution)

**Coding Exercise**: Write a Python script that imports `AdaptShotConfig`, creates it with different parameters, and prints its frozen fields. Try to mutate a field — observe the error.

**Project Challenge**: Add a new config field `inference_timeout_ms: float = 5000.0` to `AdaptShotConfig` with validation in `__post_init__`. Update the docstring. Write a test in a temporary file that validates the field.

**Research Reading**: Read AdaptShot's `README.md`, `ROADMAP.md`, and `research/v0.2.1_ROADMAP.md`. Understand the project's current state and future direction.

**Interview Question**: "Explain AdaptShot's architecture to me. What are the subsystems and how do they communicate?"

**Reflection**: What surprised you about the codebase? What module seems most intimidating?

**Next-Day Prep**: Review basic linear algebra — vectors, dot products, norms, matrix multiplication.

---

### Day 2 — Linear Algebra for ML Engineers

**Learning Objectives**: Master the vector and matrix operations used in every line of AdaptShot.

**Theory**: Why linear algebra is the language of ML. Every image becomes a vector. Every backbone produces a vector. Every comparison is a linear algebra operation.

**Mathematics**:
- Vectors in \( \mathbb{R}^D \) — what is a 512-dimensional embedding?
- Dot product \( \mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i \) — geometric intuition (projection, angle)
- Euclidean norm (L2): \( \|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2} \)
- Cosine similarity: \( \cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} \)
- Broadcasting in NumPy — how `[D] @ [N, D].T` produces `[N]`
- Matrix-vector products, matrix-matrix products, transposes
- Why L2-normalization enables cosine similarity via dot product

**Algorithms**: Binary search (searchsorted for calibration bins), argmin/argmax for nearest-neighbor selection.

**Software Engineering**: NumPy best practices — avoid Python loops, use broadcasting, prefer `np.dot` and `np.linalg.norm` over manual iteration. The `dtype=np.float32` everywhere rule and why.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/similarity.py` line by line. Understand `cosine_similarity_numpy`, `euclidean_distance_numpy`, `_l2_normalize_rows`, `_ensure_2d`.

**Code Reading**:
- `src/adaptshot/core/similarity.py` (full 241 lines)

**Coding Exercise**: Reimplement `cosine_similarity_numpy` from scratch without looking at the source. Then reimplement `euclidean_distance_numpy` using broadcasting. Test against the original.

**Project Challenge**: Add a `manhattan_distance_numpy` function to `similarity.py` with full type hints, docstring, and edge case handling. Write corresponding tests.

**Research Reading**: "Efficient Nearest Neighbor Search in High Dimensions" — understand the curse of dimensionality and why L2-normalization helps.

**Interview Question**: "Given two 512-dimensional vectors, compute their cosine similarity by hand on a whiteboard. What does a score of -1, 0, and 1 mean geometrically?"

**Reflection**: Can you now look at any line in `similarity.py` and explain what every NumPy operation does? If not, which operations confuse you?

**Next-Day Prep**: Understand probability basics — what is a probability distribution? What is entropy?

---

### Day 3 — Probability, Entropy, and Confidence

**Learning Objectives**: Master the probability concepts behind calibration, ACT, and uncertainty.

**Theory**: Why ML models output "confidence" scores. The difference between confidence and probability. Why raw neural network outputs are NOT probabilities (they're uncalibrated logits).

**Mathematics**:
- Probability axioms: \( 0 \leq P(A) \leq 1 \), \( P(\Omega) = 1 \), additivity
- Conditional probability: \( P(A|B) = P(A \cap B) / P(B) \)
- Bayes' theorem: \( P(H|E) = P(E|H) \cdot P(H) / P(E) \)
- Softmax: \( \sigma(z)_i = e^{z_i} / \sum_j e^{z_j} \) — converting logits to probabilities
- Logits: \( \text{logit}(p) = \ln(p / (1-p)) \) — the inverse of sigmoid
- Sigmoid: \( \sigma(x) = 1 / (1 + e^{-x}) \)
- Entropy: \( H(p) = -\sum_i p_i \log p_i \) — measuring uncertainty of a distribution
- Why maximum entropy occurs at uniform distribution
- Cross-entropy loss: \( L = -\sum_i y_i \log(\hat{y}_i) \) — the standard classification loss

**Algorithms**: Softmax computation with numerical stability (subtract max). Binary search for bin assignment.

**Software Engineering**: The `CalibrationEngine` class design — sliding window pattern, online updates, grid search for temperature.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/calibration.py`. Understand temperature scaling: \( \text{conf}_{\text{cal}} = \sigma(\text{logit}(\text{conf}) / T) \). Understand ECE computation. Understand the difference between ECE and debiased ECE.

**Code Reading**:
- `src/adaptshot/core/calibration.py` (full 305 lines)

**Coding Exercise**: Implement temperature scaling from scratch. Given a list of (confidence, correct) pairs, grid-search the temperature T that minimizes ECE. Then implement ECE computation manually (no looking at the source).

**Project Challenge**: Add a `compute_mce` (Maximum Calibration Error) method to `CalibrationEngine`. MCE = max over bins of |avg_acc - avg_conf|. Include tests.

**Research Reading**: Guo et al. (2017) "On Calibration of Modern Neural Networks" — the foundational paper on temperature scaling. Read sections 1-4.

**Interview Question**: "A model predicts class A with 90% confidence. After temperature scaling with T=2.0, what happens to the confidence? Why might this be desirable?"

**Reflection**: Do you understand why a model can be "overconfident"? Can you explain temperature scaling to someone who knows basic algebra?

**Next-Day Prep**: Understand k-NN algorithm and distance metrics. Why is k-NN the heart of few-shot learning?

---

### Day 4 — Nearest Neighbor Search and Distance Metrics

**Learning Objectives**: Master nearest-neighbor algorithms that power every AdaptShot prediction.

**Theory**: The fundamental assumption of few-shot learning — similar images are close in embedding space. The k-NN algorithm. The bias-variance tradeoff in choosing k.

**Mathematics**:
- Minkowski distance: \( d(\mathbf{x}, \mathbf{y}) = (\sum_i |x_i - y_i|^p)^{1/p} \)
  - p=1: Manhattan distance
  - p=2: Euclidean distance
  - p=∞: Chebyshev distance
- Cosine distance: \( d_{\cos} = 1 - \cos(\theta) \)
- Why Euclidean on L2-normalized vectors ≈ angular distance
- Distance-to-confidence mapping: \( \text{conf} = 1 / (1 + d) \)

**Algorithms**:
- k-NN: O(N·D) per query (brute force)
- KD-Trees: O(log N) for low dimensions, degrades in high D
- Ball Trees: better for high dimensions than KD-trees
- FAISS: approximate nearest neighbor (ANN) with IVF indices
- When to use exact vs approximate search
- LSH (Locality-Sensitive Hashing): random projections for approximate search

**Software Engineering**: The `find_nearest_neighbor` and `find_nearest_prototype` functions. Graceful fallback from FAISS to NumPy. The `use_faiss` toggle pattern.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/similarity.py` again, now focusing on `find_nearest_neighbor`, `find_nearest_prototype`, and FAISS integration.

**Code Reading**:
- `src/adaptshot/core/similarity.py` (review lines 143-241)
- `src/adaptshot/training/up_ugf.py` (lines 89-123 for LSH implementation)

**Coding Exercise**: Implement k-NN search from scratch. Given a query vector and N support vectors, return the k nearest neighbors sorted by distance. Then implement a weighted k-NN where closer neighbors have higher vote weight.

**Project Challenge**: Add `find_k_nearest_neighbors` (returns top-k, not just top-1) to `similarity.py`. It should support both cosine and Euclidean, both FAISS and NumPy paths. Write tests.

**Research Reading**: Johnson et al. (2019) "Billion-scale similarity search with GPUs" — the FAISS paper. Focus on the IVF index structure.

**Interview Question**: "Why does KD-tree performance degrade in 512 dimensions? What is the curse of dimensionality and how does FAISS work around it?"

**Reflection**: Can you implement nearest-neighbor search in pure Python without NumPy? How much slower would it be?

**Next-Day Prep**: Think about prototypes — why use class means instead of all support examples?

---

### Day 5 — Prototypical Networks: The Core Algorithm

**Learning Objectives**: Master prototypical networks — the mathematical heart of AdaptShot.

**Theory**: Prototypical Networks (Snell et al., 2017). Each class is represented by a single prototype vector — the mean of its support embeddings. Classification: assign query to nearest prototype. Why this works: the embedding function learns a metric space where Euclidean distance encodes class membership.

**Mathematics**:
- Prototype: \( \mathbf{c}_k = \frac{1}{|S_k|} \sum_{\mathbf{x}_i \in S_k} f_\phi(\mathbf{x}_i) \)
- Distance-based softmax: \( p(y=k|\mathbf{x}) = \frac{\exp(-d(f_\phi(\mathbf{x}), \mathbf{c}_k))}{\sum_{k'} \exp(-d(f_\phi(\mathbf{x}), \mathbf{c}_{k'}))} \)
- Why mean pooling: it's equivalent to assuming isotropic Gaussian class distributions with equal variance
- The connection to Gaussian mixture models
- Prototype margin: gap between nearest and second-nearest class — a confidence proxy

**Algorithms**: Mean computation with streaming/online update (useful for human feedback additions). Running mean: \( \mu_{n+1} = \mu_n + (x_{n+1} - \mu_n) / (n+1) \).

**Software Engineering**: The `compute_class_prototypes` function — O(N) single-pass grouping. The `_rebuild_prototypes` pattern in `FewShotLearner`. When to rebuild vs incrementally update.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/similarity.py`'s `compute_class_prototypes` function. Read `FewShotLearner._rebuild_prototypes` and `FewShotLearner._compute_all_prototype_distances`.

**Code Reading**:
- `src/adaptshot/core/similarity.py` lines 96-141 (`compute_class_prototypes`)
- `src/adaptshot/core/learner.py` lines 1014-1028 (`_rebuild_prototypes`)
- `src/adaptshot/core/learner.py` lines 1096-1108 (`_compute_all_prototype_distances`)

**Coding Exercise**: Implement `compute_class_prototypes` from scratch. Then extend it to also compute per-class variance (spread of embeddings around the mean). Test on random data.

**Project Challenge**: Write a `prototype_quality_report` function that computes: (1) inter-class separation (min pairwise prototype distance), (2) intra-class compactness (mean distance of support examples to their prototype), (3) the ratio of these as a quality score. Add it to `similarity.py` with tests.

**Research Reading**: Snell et al. (2017) "Prototypical Networks for Few-shot Learning" — the foundational paper. Read it in full.

**Interview Question**: "Prototypical networks use one prototype per class. What happens when a class is multi-modal (e.g., 'dog' includes both chihuahuas and great danes)? How would you fix this?"

**Reflection**: Week 1 complete. Can you explain the core AdaptShot prediction pipeline end-to-end? Image → backbone → embedding → prototype distance → prediction?

**Next-Day Prep**: Review the ImageNet preprocessing pipeline. What transforms are applied to images before they enter the backbone?

---

## Week 2: Core ML Engineering

### Day 6 — Backbone Extraction and Transfer Learning

**Learning Objectives**: Understand how frozen backbones produce embeddings, ImageNet transfer learning, and the preprocessing pipeline.

**Theory**: Transfer learning — why a ResNet-18 trained on ImageNet produces useful features for crop diseases, wildlife, etc. The concept of a "frozen backbone" — weights are never updated, only embeddings are extracted. Why this works: early layers learn edges/textures, later layers learn semantic concepts.

**Mathematics**:
- Convolution basics (conceptual): sliding filters over images to detect patterns
- Pooling operations: max pooling, average pooling, global average pooling (GAP)
- ResNet architecture: skip connections, \( y = F(x) + x \), addressing vanishing gradients
- MobileNet V3: depthwise separable convolutions, squeeze-and-excitation
- ImageNet normalization: \( x' = (x - \mu) / \sigma \) with \( \mu = [0.485, 0.456, 0.406] \), \( \sigma = [0.229, 0.224, 0.225] \)

**Algorithms**: LRU caching for backbone models (`@lru_cache`). The EmbeddingCache pattern for eco-mode early exit. Preview signatures — 32x32 downsampled images as cheap similarity proxies.

**Software Engineering**: The `_get_*()` lazy import pattern — why it's essential for a torch-optional library. The `BackboneRegistry` factory pattern. The `clear_backbone_cache()` function for memory management.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/extractor.py` fully. Understand `_build_backbone`, `extract_embedding`, the eco-mode fast path, how the classification head is stripped (`.fc = nn.Identity()`).

**Code Reading**:
- `src/adaptshot/core/extractor.py` (full 274 lines)

**Coding Exercise**: Write a script that loads a ResNet-18, strips its classifier, passes an image through it, and inspects the 512-dim embedding. Print the top 10 and bottom 10 activation values. What do you notice?

**Project Challenge**: Add MobileNet V2 support to `BackboneRegistry`. Look up its output dimension, add the registry entry, update `BACKBONE_OUTPUT_DIM`. Verify extraction works.

**Research Reading**: He et al. (2016) "Deep Residual Learning for Image Recognition" — the ResNet paper. Focus on sections 1-3 (the architecture, not the training details).

**Interview Question**: "Why use a frozen backbone instead of fine-tuning it? When would fine-tuning the backbone be necessary?"

**Reflection**: What is the tradeoff between ResNet-18 (512-dim) and MobileNet V3 (576-dim)? Why might one be preferred over the other?

**Next-Day Prep**: Review the contrastive learning module. What problem does it solve that plain prototypes don't?

---

### Day 7 — Contrastive Learning and InfoNCE

**Learning Objectives**: Master contrastive prototype learning — the most algorithmically complex module in AdaptShot.

**Theory**: Why naive mean prototypes can fail: multi-modal classes, noisy embeddings, overlapping distributions. Contrastive learning pushes same-class embeddings together and different-class embeddings apart. The InfoNCE loss (Oord et al., 2018): a noise-contrastive estimation objective.

**Mathematics**:
- InfoNCE: \( L = -\frac{1}{N} \sum_i \log \frac{\exp(\text{sim}(z_i, z_{\text{pos}}) / \tau)}{\sum_j \exp(\text{sim}(z_i, z_j) / \tau)} \)
- Temperature \( \tau \): controls concentration of the distribution. Lower \( \tau \) = harder assignments
- Gradient of InfoNCE with respect to embeddings (the backward pass derivation)
- Momentum (SGD with momentum): \( v_t = \beta v_{t-1} - \eta \nabla L \), \( \theta_t = \theta_{t-1} + v_t \)
- He initialization: \( W \sim \mathcal{N}(0, \sqrt{2 / n_{\text{in}}}) \) for ReLU activations
- Hard negative mining: up-weighting close negatives in the loss

**Algorithms**: Mini-batch SGD with momentum. Early stopping criteria. L2 normalization for cosine-similarity based contrastive learning.

**Software Engineering**: The 2-layer MLP projection head pattern. Training loop with forward pass (project), compute loss, backward pass (manual gradient computation through the projection layers). The `_train_projection_head` function is a complete mini deep learning framework in NumPy.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/contrastive.py` in its entirety. This is the most mathematically dense module. Understand every line of `_compute_infonce_loss` and `_train_projection_head`.

**Code Reading**:
- `src/adaptshot/core/contrastive.py` (full 513 lines)

**Coding Exercise**: Implement InfoNCE loss from scratch. Given N normalized embeddings and their labels, compute the loss and verify that randomly initialized embeddings produce a baseline loss of approximately log(N).

**Project Challenge**: In `contrastive.py`, the hard negative weighting modifies the softmax probabilities. Trace through the code and verify that the gradient computation is correct. Write a test that verifies the gradient numerically (finite differences).

**Research Reading**: Chen et al. (2020) "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR) — sections 1-3. Understand why contrastive learning works without labels.

**Interview Question**: "Explain the difference between contrastive learning for representation learning (SimCLR) and contrastive prototype refinement (AdaptShot). Why does AdaptShot only refine prototypes, not embeddings?"

**Reflection**: Can you derive the gradient of InfoNCE loss by hand on a whiteboard?

**Next-Day Prep**: Understand ACT (Adaptive Confidence Thresholding). Why can't we just use a fixed threshold for all predictions?

---

### Day 8 — Adaptive Confidence Thresholding (ACT)

**Learning Objectives**: Master the ACT engine — online threshold adaptation with mean reversion.

**Theory**: Why a fixed threshold (e.g., 0.5) is insufficient: different classes have different difficulty levels, and data distributions drift over time. ACT adapts per-class thresholds based on correction history. The tradeoff between false acceptances (costly mistakes) and false rejections (unnecessary human queries).

**Mathematics**:
- Threshold update: \( \Delta = \eta \cdot (\text{incorrect\_rate} - \text{correct\_rate}) + \mu \cdot (\tau_{\text{base}} - \tau) \)
- The first term: error signal — raise threshold when mistakes are common, lower when they're rare
- The second term: mean reversion — thresholds drift back to baseline, preventing runaway
- Exponential moving average (EMA): \( s_t = \alpha \cdot x_t + (1-\alpha) \cdot s_{t-1} \)
- Why v0.2.0 fixed the asymmetric update bug: the old formula created permanent downward bias

**Algorithms**: Online learning (update per prediction). Clamping to valid ranges. The per-class state dictionary pattern.

**Software Engineering**: The `should_accept` method design — returns both a boolean and an action string. The `get_threshold` / `get_all_thresholds` snapshot pattern. Dynamic class expansion (handling new classes discovered through feedback).

**AdaptShot Deep Dive**: Read `src/adaptshot/core/act.py` fully. Understand the v0.2.0 fix documented in the comments — trace through the math of the old vs new update rule.

**Code Reading**:
- `src/adaptshot/core/act.py` (full 153 lines)

**Coding Exercise**: Implement ACT from scratch. Initialize with base_threshold=0.65. Simulate 100 predictions with varying incorrect rates. Plot how the threshold adapts. Verify that mean reversion prevents the threshold from drifting to extreme values.

**Project Challenge**: Add a `decay_untouched_classes` method to `ACTEngine` that slowly reverts thresholds of classes that haven't been seen recently (use a simple time-since-last-access heuristic). Write tests.

**Research Reading**: Geifman & El-Yaniv (2017) "Selective Classification for Deep Neural Networks" — the concept of "reject option" in classification.

**Interview Question**: "Why does ACT use per-class thresholds instead of a global threshold? What would happen if two classes had very different inherent difficulties?"

**Reflection**: Can you identify a scenario where ACT's mean reversion would cause a problem? How would you fix it?

**Next-Day Prep**: Understand how the calibration engine and ACT work together. Why do we need both?

---

### Day 9 — End-to-End Prediction Pipeline

**Learning Objectives**: Trace a single `predict()` call through every subsystem.

**Theory**: The prediction pipeline: image → extract embedding → find nearest neighbor/prototype → calibrate confidence → ACT gate → OOD check → conformal set → uncertainty report → neighbors → structured result. Why this ordering matters — calibration happens before ACT, OOD overrides ACT, conformal wraps everything.

**Mathematics**: Review the confidence computation chain: raw similarity → distance-to-confidence mapping → temperature scaling → ACT penalty → OOD penalty → final confidence.

**Algorithms**: The full prediction algorithm as a DAG (directed acyclic graph) of transformations.

**Software Engineering**: The `PredictionResult` dataclass — why return a dataclass instead of a dict. The `_calibrate_or_raise` graceful fallback pattern (v0.2.0 improvement). The `_coerce_label` and `_label_key` normalization.

**AdaptShot Deep Dive**: Read `learner.py` `predict()` method (lines 289-426) line by line. Trace every branch: nearest_neighbor vs prototypical vs contrastive. Understand what happens when calibration isn't ready, when OOD is detected, when contrastive mode is active.

**Code Reading**:
- `src/adaptshot/core/learner.py` lines 289-426 (`predict`)
- `src/adaptshot/core/learner.py` lines 99-117 (`PredictionResult`)
- `src/adaptshot/core/learner.py` lines 986-1003 (`_calibrate_or_raise`, `_raw_to_unit_interval`)

**Coding Exercise**: Build a "mini predict" function that: takes a query embedding + support embeddings/labels → returns predicted label, raw confidence, calibrated confidence, neighbor index, and distance. Implement in ~50 lines.

**Project Challenge**: In `predict()`, add a `return_intermediates=True` parameter that dumps all intermediate values (raw embedding, all distances, pre/post calibration confidence, ACT threshold used) into a debug dict in `PredictionResult`. Write tests.

**Research Reading**: Re-read the Prototypical Networks paper — now with full understanding of every component.

**Interview Question**: "A bug report says 'sometimes ACT rejects predictions that OOD would have accepted.' Walk me through the code that handles this. Is it a bug or by design?"

**Reflection**: Can you draw the full prediction pipeline from memory, including every data transformation?

**Next-Day Prep**: What are iterators, generators, and list comprehensions doing under the hood? How does Python's memory model affect ML code?

---

### Day 10 — Python Internals for ML Engineers

**Learning Objectives**: Master Python internals that affect performance-critical ML code.

**Theory**: Python's memory model — reference counting, garbage collection. Why `np.array(list_of_arrays)` is slow. Why `np.float32` everywhere in AdaptShot.

**Mathematics**: None — systems day.

**Algorithms**: Hash tables (dict internals). List growth strategy (over-allocation). The iterator protocol.

**Software Engineering**:
- List comprehensions vs generator expressions vs map/filter
- `__slots__` for memory-efficient classes
- `@dataclass` vs `NamedTuple` vs plain class
- `functools.lru_cache` internals
- Context managers (`with` statement) — how `tempfile` and `try/finally` work in `save()`
- The GIL (Global Interpreter Lock) and why it doesn't matter for NumPy (NumPy releases the GIL)
- Profiling with `cProfile`, `memory_profiler`, `tracemalloc`
- Why `np.zeros(N, dtype=np.float32)` vs `np.zeros(N)` matters (float64 default)

**AdaptShot Deep Dive**: Profile `predict()` using cProfile. Identify the slowest operations. Profile memory with tracemalloc. Verify the <250MB budget.

**Code Reading**:
- `src/adaptshot/utils/profiling.py` (if it exists)
- `benchmarks/energy_profile.py`
- `src/adaptshot/core/learner.py` lines 635-706 (`save` method — the atomic write pattern)

**Coding Exercise**: Benchmark 5 different ways to build a list of 1000 numpy arrays (list append, preallocate, fromiter, etc.). Which is fastest? Which uses least memory?

**Project Challenge**: Profile `load_support_images` with 100 images. Identify memory hotspots. If any operation allocates >10MB, document it and propose a fix.

**Research Reading**: "Python's Memory Management" — Python docs + realpython.com article on Python memory.

**Interview Question**: "Why does `arr = np.array([np.zeros(512) for _ in range(1000)])` use nearly 16MB of memory (instead of expected ~2MB)?"

**Reflection**: Week 2 complete. Can you explain how every line of `predict()` works, from image input to PredictionResult output?

**Next-Day Prep**: Review uncertainty estimation. What does it mean for a model to "know what it doesn't know"?

---

## Week 3: Advanced Algorithms

### Day 11 — Multi-Signal Uncertainty: Epistemic & Aleatoric

**Learning Objectives**: Master epistemic (model) and aleatoric (data) uncertainty estimation.

**Theory**: Why uncertainty matters for trustworthy AI. The difference between epistemic uncertainty ("I don't know because I haven't seen enough data") and aleatoric uncertainty ("I don't know because the data is inherently ambiguous"). Examples: a blurry photo of a dog (aleatoric) vs a photo of a rare species never in training (epistemic).

**Mathematics**:
- Epistemic: perturbation sensitivity — \( U_{\text{epi}} = \text{Var}(\{f(\mathbf{x} + \epsilon_i)\}_{i=1}^M) \), normalized to [0, 1]
- Why perturbation of embeddings proxies for MC Dropout
- Aleatoric: k-NN entropy — \( H = -\sum_c p_c \log p_c \) where \( p_c \) is the weighted vote of class c among k neighbors
- Normalization: divide by \( \log(K) \) (max entropy for K classes)
- Composite uncertainty: \( U = \frac{w_e U_e + w_a U_a + w_d U_d}{w_e + w_a + w_d} \)

**Algorithms**: k-NN entropy — O(N·D + k log k) per query. Perturbation sampling — O(M·D) where M is number of perturbations.

**Software Engineering**: The `UncertaintyQuantifier` class design — mode-gated computation (`"mcdropout"`, `"entropy"`, `"mahalanobis"`, `"ensemble"`). The `UncertaintyReport` dataclass with `to_dict()` serialization.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/uncertainty.py` lines 322-455 (epistemic + aleatoric). Understand `estimate_epistemic`, `compute_knn_entropy`, and the composite score calculation.

**Code Reading**:
- `src/adaptshot/core/uncertainty.py` lines 322-526 (epistemic, aleatoric, quantify)

**Coding Exercise**: Implement k-NN entropy from scratch. Test with: two clusters far apart (should give low entropy), one cluster with mixed labels (should give high entropy).

**Project Challenge**: Add a `calibrate_uncertainty` method to `UncertaintyQuantifier` that learns optimal weights (w_e, w_a, w_d) from labeled calibration data using grid search to maximize AUROC for OOD detection. Write tests.

**Research Reading**: Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" — the paper that defined the epistemic/aleatoric taxonomy.

**Interview Question**: "A prediction has high epistemic uncertainty but low aleatoric uncertainty. What does this mean in practice? What should the system do?"

**Reflection**: What are the limitations of perturbation-based epistemic uncertainty? When would MC Dropout be better?

**Next-Day Prep**: Review covariance matrices and Mahalanobis distance. How does it differ from Euclidean distance?

---

### Day 12 — Mahalanobis Distance and OOD Detection

**Learning Objectives**: Master Mahalanobis distance and out-of-distribution detection.

**Theory**: Why Euclidean distance fails for OOD detection — it treats all dimensions equally, ignoring correlations. Mahalanobis distance accounts for the covariance structure of in-distribution data: \( D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)} \). In-distribution samples have small Mahalanobis distance; OOD samples have large distance.

**Mathematics**:
- Covariance matrix: \( \Sigma = \frac{1}{n-1} (\mathbf{X} - \mu)^T (\mathbf{X} - \mu) \)
- Mahalanobis distance: shape-aware distance — a unit circle in Euclidean becomes an ellipse in Mahalanobis
- Why \( \Sigma^{-1} \) fails in few-shot: when n < D, \( \Sigma \) is rank-deficient (singular)
- Shrinkage estimation (Ledoit-Wolf): \( \Sigma_{\text{shrunk}} = (1-\alpha) S + \alpha \cdot \text{diag}(S) \)
- Shrinkage factor: \( \alpha = D / (D + n) \) — more shrinkage when fewer samples
- Pseudoinverse as fallback: \( \Sigma^+ \) via SVD
- OOD threshold: percentile of in-distribution Mahalanobis distances

**Algorithms**: Covariance computation (O(N·D²)). Matrix inversion (O(D³)). Cholesky decomposition for efficient Mahalanobis (optional). Quantile computation via `np.percentile`.

**Software Engineering**: The `fit_class_distributions` method — per-class Gaussian fitting with shrinkage. The `_compute_ood_threshold` method. The class-conditional vs global OOD decision.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/uncertainty.py` lines 138-319 (Mahalanobis + OOD). Understand every line of the shrinkage logic — this is where the mathematical rubber meets the road.

**Code Reading**:
- `src/adaptshot/core/uncertainty.py` lines 138-319

**Coding Exercise**: Implement Mahalanobis distance from scratch. Generate 2D data from two correlated Gaussians. Compute Euclidean and Mahalanobis distances to each cluster center. Visualize the decision boundaries. Show a case where Euclidean makes a wrong decision but Mahalanobis gets it right.

**Project Challenge**: In `uncertainty.py`, the `_compute_ood_threshold` uses a simple percentile. Enhance it with an adaptive threshold that updates online as new in-distribution data arrives (running mean/variance of distances). Write tests.

**Research Reading**: Lee et al. (2018) "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" — the Mahalanobis OOD paper.

**Interview Question**: "With only 5 support examples per class in 512-dimensional space, what goes wrong with Mahalanobis distance? How does AdaptShot's shrinkage fix address this?"

**Reflection**: Can you explain to a beginner why \( 1/(x^T \Sigma^{-1} x) \) is not simply a "scaled Euclidean distance"?

**Next-Day Prep**: Understand conformal prediction. What does it mean to have "distribution-free coverage guarantees"?

---

### Day 13 — Conformal Prediction: Theory and Implementation

**Learning Objectives**: Master conformal prediction — distribution-free uncertainty sets with coverage guarantees.

**Theory**: Conformal prediction provides prediction sets (not single labels) with a mathematical guarantee: \( P(Y_{\text{test}} \in \hat{C}(X_{\text{test}})) \geq 1 - \alpha \), assuming exchangeability of calibration and test data. This is the only uncertainty framework that provides finite-sample, distribution-free guarantees.

**Mathematics**:
- Exchangeability: the joint distribution of \( (Z_1, ..., Z_n) \) is invariant under permutation
- Nonconformity score: \( s_i = A(Z_i) \) — measures how "unusual" a data point is
- Conformal p-value: \( p = |\{i: s_i \geq s_{n+1}\}| / (n+1) \)
- Prediction set: \( \hat{C}_{\alpha} = \{y: p(y) > \alpha\} \)
- Finite-sample correction: \( \hat{q} = \text{quantile}(s_{1:n}, \lceil (n+1)(1-\alpha) \rceil / n) \)
- Split conformal: calibrate on held-out set, predict on test
- Cross conformal: k-fold averaging of quantiles for more stable estimates
- Why the +1 correction matters: guarantees the coverage bound

**Algorithms**: Quantile computation with finite-sample correction. Leave-one-out conformal calibration. Rolling calibration buffer management (sliding window).

**Software Engineering**: The `ConformalEngine` class — calibration buffer management, quantile computation, prediction set generation. The `ConformalPredictionSet` dataclass. The `softmax_nonconformity` and `distance_nonconformity` score functions.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/conformal.py` fully. Understand the LOO calibration in `FewShotLearner._self_calibrate_conformal`. Trace how conformal sets integrate into `predict()`.

**Code Reading**:
- `src/adaptshot/core/conformal.py` (full 437 lines)
- `src/adaptshot/core/learner.py` lines 1153-1227 (`_self_calibrate_conformal`)

**Coding Exercise**: Implement split-conformal prediction from scratch for a simple regression problem. Generate synthetic data, calibrate on half, test on the other half. Verify that empirical coverage ≥ 1-α for α=0.1.

**Project Challenge**: In `ConformalEngine`, implement `predict_set_class_conditional` with per-class quantiles. This exists but has bugs — find and fix them. Write comprehensive tests verifying coverage per class.

**Research Reading**: Vovk et al. (2005) "Algorithmic Learning in a Random World" — the conformal prediction book. Read chapters 1-2. Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction" — the accessible tutorial.

**Interview Question**: "Why is leave-one-out calibration important for few-shot conformal prediction? What would happen if we calibrated on the same data used to compute prototypes?"

**Reflection**: What are the practical limitations of conformal prediction? When might the exchangeability assumption be violated?

**Next-Day Prep**: Understand explainability methods. Why does transparency matter for human-in-the-loop systems?

---

### Day 14 — Explainability: Attributions and Counterfactuals

**Learning Objectives**: Master model-agnostic explainability for few-shot predictions.

**Theory**: Why explainability matters for trust — users need to understand why the model made a prediction before they can provide meaningful corrections. Three complementary methods: feature attribution (which support examples influenced the prediction?), confidence decomposition (how was the confidence score computed?), and counterfactual analysis (what would change the prediction?).

**Mathematics**:
- Inverse distance weighting: \( w_i = (d_i + \epsilon)^{-1} / \sum_j (d_j + \epsilon)^{-1} \)
- Counterfactual margin: \( \text{margin} = d_{\text{cf}} - d_{\text{current}} \)
- Swap required: \( \max(0, -\text{margin}) \)
- Feature-level saliency: \( |\mathbf{x}_{\text{query}} - \mathbf{c}_{\text{predicted}}| \)

**Algorithms**: Top-k selection via argsort. Pairwise distance computation. Class-conditional minimum distance.

**Software Engineering**: The `ExplainabilityEngine` with four explanation methods. The structured `ExplanationResult`, `FeatureAttribution`, `ConfidenceDecomposition`, `Counterfactual` dataclasses. The human-readable summary generation.

**AdaptShot Deep Dive**: Read `src/adaptshot/core/explain.py` fully. Understand every explanation method and how they combine.

**Code Reading**:
- `src/adaptshot/core/explain.py` (full 587 lines)

**Coding Exercise**: Implement counterfactual explanation from scratch. Given query embedding, support set, and predicted label, find the nearest alternative class and compute how much closer the query would need to be to flip the prediction.

**Project Challenge**: Add a `diverse_counterfactuals` method that returns the top-k alternative classes with their margins. This helps users understand not just the closest alternative but all plausible alternatives. Write tests.

**Research Reading**: Wachter et al. (2017) "Counterfactual Explanations without Opening the Black Box" — the foundational counterfactual paper. Ribeiro et al. (2016) "Why Should I Trust You?" — the LIME paper.

**Interview Question**: "A user sees 'Predicted: Tomato Blight (confidence 0.87). Most influenced by support example #42.' What additional information would help the user decide whether to trust this prediction?"

**Reflection**: How would you explain AdaptShot's prediction to a farmer who has never used ML? What matters to them vs to a researcher?

**Next-Day Prep**: Review the `FewShotLearner.save()` and `load()` methods. How does checkpoint integrity work?

---

### Day 15 — Persistence, Integrity, and Migration

**Learning Objectives**: Master state persistence with integrity verification and schema migration.

**Theory**: Why save/load matters for real-world deployment — models accumulate state (calibration history, buffer contents, learned thresholds) that must survive restarts. Atomic writes: write to temp file, fsync, then os.replace — prevents corruption on crash. SHA-256 checksums for tamper detection.

**Mathematics**:
- SHA-256: cryptographic hash function — collision-resistant, avalanche effect
- JSON serialization constraints: no NaN, no Inf, no numpy arrays directly
- Schema versioning and migration: transforming old state formats to new ones

**Algorithms**: Hash chaining for integrity verification. Deterministic JSON serialization.

**Software Engineering**: The `save()`/`load()` pattern — three files (JSON state, NPY embeddings, PT model head). The atomic write pattern with `tempfile.NamedTemporaryFile` + `os.replace`. Schema migration: `migrate_v0_1_0_to_v0_1_1`. The `_build_integrity_payload` + `_validate_and_normalize_state` pattern.

**AdaptShot Deep Dive**: Read `learner.py` `save()` and `load()` methods fully. Trace the integrity verification chain: config → JSON → SHA-256, embeddings → binary → SHA-256, combined → checksum.

**Code Reading**:
- `src/adaptshot/core/learner.py` lines 635-776 (`save`, `load`)
- `src/adaptshot/core/learner.py` lines 1320-1483 (helper methods for save/load)
- `src/adaptshot/utils/migrations.py`

**Coding Exercise**: Implement atomic file writes from scratch. Write to temp, fsync, atomic rename. Test that the original file is never corrupted even if the process is killed mid-write (simulate with a small delay).

**Project Challenge**: The `save()` method currently saves all embeddings as one .npy file. For large support sets (>1000), implement memory-mapped saving using `np.memmap`. This requires changes to both save() and load(). Write tests.

**Research Reading**: "How SQLite Uses Atomic Commit" — a classic piece on file integrity patterns.

**Interview Question**: "A checkpoint file passes SHA-256 verification but produces different predictions after loading. What could be wrong?"

**Reflection**: Week 3 complete. Can you implement every algorithm in AdaptShot from scratch using only NumPy?

**Next-Day Prep**: Understand the human-in-the-loop pipeline. How do corrections flow from user → router → calibration → fine-tuning?

---

## Week 4: Human-in-the-Loop & Continual Learning

### Day 16 — Feedback Router: Human Corrections as First-Class Signals

**Learning Objectives**: Master the feedback routing system that makes AdaptShot truly human-in-the-loop.

**Theory**: Human corrections are not afterthoughts — they're primary learning signals. Each correction updates: calibration window, ACT thresholds, support buffer, and triggers fine-tuning. The correction pipeline: validate → route → update calibration → append to buffer → rebuild prototypes → check buffer capacity → trigger fine-tuning if threshold met.

**Mathematics**: Confidence-weighted corrections: high-confidence human feedback (1.0) has more impact than uncertain feedback (0.0). The FIFO buffer eviction policy (v0.1) vs UP-UGF scoring (v0.2).

**Algorithms**: Circular buffer (FIFO). Trigger threshold pattern (accumulate N corrections, then act).

**Software Engineering**: The `Correction` dataclass — structured feedback event. The `FeedbackRouter` class — decouples feedback ingestion from processing. The `finetune_fn` callback pattern — dependency injection for testability.

**AdaptShot Deep Dive**: Read `src/adaptshot/training/feedback_router.py` and `FewShotLearner.correct()` fully.

**Code Reading**:
- `src/adaptshot/training/feedback_router.py` (full 140 lines)
- `src/adaptshot/core/learner.py` lines 480-577 (`correct`)
- `src/adaptshot/core/learner.py` lines 579-624 (`correct_comparative`)

**Coding Exercise**: Implement a minimal feedback router from scratch. Store corrections in a list. When 5 accumulate, compute the most common corrected label and "retrain" (just update a prototype).

**Project Challenge**: Add `correct_batch` method to `FewShotLearner` that accepts multiple (image_path, true_label) pairs and routes them efficiently (single prototype rebuild after all corrections). Write tests.

**Research Reading**: Amershi et al. (2014) "Power to the People: The Role of Humans in Interactive Machine Learning" — the HITL foundational paper.

**Interview Question**: "A user provides 10 corrections, but only 3 are correct (the rest are user errors). How should AdaptShot handle noisy human feedback? What mechanisms exist or should exist?"

**Reflection**: How does AdaptShot's feedback loop compare to active learning? To reinforcement learning from human feedback (RLHF)?

**Next-Day Prep**: Understand EWC (Elastic Weight Consolidation). Why does fine-tuning on new data cause catastrophic forgetting?

---

### Day 17 — Elastic Weight Consolidation (EWC)

**Learning Objectives**: Master continual learning via Fisher Information regularization.

**Theory**: Catastrophic forgetting — when a neural network fine-tuned on new data loses performance on old data. EWC solution: add a penalty term \( L_{\text{EWC}} = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2 \) that anchors important parameters near their old values. Fisher Information Matrix F measures parameter importance: parameters with high F were crucial for old tasks.

**Mathematics**:
- Fisher Information: \( F = \mathbb{E}[(\nabla_\theta \log p(y|x,\theta))^2] \) — expected squared gradient
- Diagonal approximation: \( F_i \approx \frac{1}{N} \sum_n (\nabla_{\theta_i} \log p(y_n|x_n,\theta))^2 \)
- EWC loss: \( L(\theta) = L_{\text{task}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^{\text{old}})^2 \)
- Why diagonal: O(P) storage vs O(P²) for full Fisher
- Why AdaptShot uses head-only EWC: ~2K parameters (512 × 5 classes) instead of ~11M (full ResNet-18)

**Algorithms**: Gradient computation and squaring. SGD with L2 regularization. Adam optimizer.

**Software Engineering**: The `CAEWCFinetuner` class — Fisher computation, confidence-weighted EWC penalty, standard fine-tuning fallback. The `update_fisher` + `finetune` two-step pattern. Why Fisher is computed on support set (old knowledge), not correction data.

**AdaptShot Deep Dive**: Read `src/adaptshot/training/finetune.py` fully. Understand both EWC and standard fine-tuning paths.

**Code Reading**:
- `src/adaptshot/training/finetune.py` (full 210 lines)

**Coding Exercise**: Implement diagonal EWC from scratch for a simple logistic regression model. Train on task A, compute Fisher, train on task B with EWC penalty. Measure performance on both tasks. Compare with naive fine-tuning (no EWC).

**Project Challenge**: In `CAEWCFinetuner`, add an `online_fisher_update` method that updates the Fisher diagonals incrementally as new support examples arrive (running average of squared gradients). Write tests.

**Research Reading**: Kirkpatrick et al. (2017) "Overcoming Catastrophic Forgetting in Neural Networks" — the EWC paper. Read it fully.

**Interview Question**: "Why does AdaptShot's EWC operate only on the classification head? What would change if we fine-tuned the entire ResNet-18 backbone with EWC?"

**Reflection**: Can you derive the Fisher Information Matrix formula from first principles (starting from KL divergence between old and new parameter distributions)?

**Next-Day Prep**: Understand UP-UGF buffer pruning. Why can't we just keep all support examples forever?

---

### Day 18 — Buffer Management (UP-UGF)

**Learning Objectives**: Master uncertainty-guided buffer pruning for bounded memory.

**Theory**: The buffer capacity constraint (max 100 examples, <250MB RAM). Naive FIFO eviction loses the most informative examples. UP-UGF assigns a utility score to each example: Score(e) = (1-U(e))^w_unc × exp(-λ·Δt)^w_rec × (1-max_sim_same_class)^w_red. High-uncertainty, recently-used, diverse examples are retained.

**Mathematics**:
- Uncertainty component: \( (1 - u)^{w_{\text{unc}}} \) — prefer informative/uncertain examples
- Recency component: \( \exp(-\lambda \cdot \Delta t)^{w_{\text{rec}}} \) — exponential decay of utility
- Redundancy component: \( (1 - \max\_\text{sim}_{\text{same}})^{w_{\text{red}}} \) — penalize duplicates
- Multiplicative fusion: all three must be high for retention
- LSH (Locality-Sensitive Hashing): random projection → hash bits → collision counting for approximate similarity

**Algorithms**: LSH for O(N·D·log N) approximate redundancy instead of O(N²·D) exact. Priority queue (argsort) for top-K selection. Sliding window for recency.

**Software Engineering**: The `UPUGFPruner` class — `compute_scores` + `prune` methods. The O(N²) → LSH fallback at N>100. The deterministic FIFO fallback on pruning failure.

**AdaptShot Deep Dive**: Read `src/adaptshot/training/up_ugf.py` fully. Understand the LSH implementation — this is a clever performance optimization.

**Code Reading**:
- `src/adaptshot/training/up_ugf.py` (full 161 lines)
- `src/adaptshot/core/learner.py` lines 1614-1643 (`_apply_buffer_management`)

**Coding Exercise**: Implement UP-UGF scoring from scratch. Test with synthetic data: create 150 embeddings with varying uncertainty, recency, and redundancy. Verify that pruning retains the "right" 100.

**Project Challenge**: The LSH fallback uses a fixed random seed (42). Make it deterministic but configurable. Add a `lsh_random_seed` parameter to `UPUGFPruner`. Write tests verifying determinism.

**Research Reading:** Review the LSH literature — Indyk & Motwani (1998) "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality".

**Interview Question:** "UP-UGF uses a multiplicative score. What would happen if we used an additive score instead? Under what conditions would the two diverge?"

**Reflection:** How would you evaluate whether UP-UGF is actually better than FIFO? Design an experiment.

**Next-Day Prep:** Understand the integration between feedback, fine-tuning, and buffer management. How do they work together in `correct()`?

---

### Day 19 — The Complete correct() Pipeline

**Learning Objectives:** Trace a single `correct()` call through every subsystem — feedback routing, calibration update, buffer append, prototype rebuild, OOD update, buffer pruning, conformal update, fine-tuning trigger.

**Theory:** The full HITL loop: user provides (image, true_label, confidence_weight) → nearest neighbor check → create Correction → route to FeedbackRouter → update calibration window → append to similarity buffer → rebuild prototypes → update OOD threshold → apply buffer management → update conformal calibration → check fine-tuning trigger. Why the ordering of these steps matters.

**Mathematics:** Review: temperature grid search (cross-entropy minimization), ECE computation, prototype mean update, OOD quantile update, conformal quantile update.

**Algorithms:** The full correct() pipeline as a state machine.

**Software Engineering:** Error handling in the pipeline — non-critical subsystems (buffer pruning) must not crash the pipeline. The `try/except` around `_apply_buffer_management` with FIFO fallback. The comparative feedback extension (`correct_comparative`).

**AdaptShot Deep Dive:** Read `FewShotLearner.correct()` (lines 480-577) and trace every branch. Then read `correct_comparative` (lines 579-624) and understand how ordinal supervision maps to standard correction.

**Code Reading:**
- `src/adaptshot/core/learner.py` lines 480-624

**Coding Exercise:** Simulate a complete HITL loop: initialize learner → predict 10 images → provide 3 corrections → predict again → verify that calibration improved (lower ECE) and buffer was pruned (size ≤ capacity).

**Project Challenge:** The `correct()` method requires re-extracting the embedding for the corrected image, even though it was already extracted during `predict()`. Add a `correct_from_embedding` method that accepts a pre-extracted embedding. Add an optional `embedding` parameter to `correct()`. Write tests.

**Research Reading:** Settles (2009) "Active Learning Literature Survey" — understand how HITL relates to active learning.

**Interview Question:** "Correct() updates calibration, prototypes, OOD threshold, and conformal scores. Which of these updates are commutative (order doesn't matter) and which are not? Why?"

**Reflection:** Can you design an experiment to measure how much human feedback improves AdaptShot's accuracy? What metrics would you track?

**Next-Day Prep:** Review the testing framework. What makes a good test for ML code?

---

### Day 20 — Testing ML Systems

**Learning Objectives:** Master testing strategies for ML libraries.

**Theory:** The testing pyramid for ML: unit tests (individual functions), integration tests (subsystem interactions), system tests (end-to-end pipelines), property tests (mathematical invariants), regression tests (bug fixes), determinism tests (bit-exact reproducibility). Why ML testing is harder than traditional software testing — non-determinism, floating-point tolerance, statistical claims.

**Mathematics:** Floating-point comparison with absolute/relative tolerance. Property-based testing: invariant properties that must always hold (e.g., ECE ∈ [0, 1], confidence ∈ [0, 1], OOD score ≥ 0).

**Algorithms:** Deterministic seed management. Test fixtures and mocking.

**Software Engineering:**
- pytest patterns: fixtures, parametrize, monkeypatch, tmp_path
- Mocking heavy dependencies (torch, FAISS) for unit tests
- Test file organization mirroring source structure
- The full quality gate: `ruff check` → `mypy --strict` → `pytest -v` → smoke test benchmark

**AdaptShot Deep Dive:** Read the test suite. Understand what each test file covers.

**Code Reading:**
- All files in `tests/`
- `.github/workflows/ci.yml`

**Coding Exercise:** Write 5 unit tests for `cosine_similarity_numpy`: empty input, single vector, known orthogonal vectors, known parallel vectors, numerical stability near zero.

**Project Challenge:** `test_learner_integration.py` has gaps. Write a complete integration test for the full `predict()` → `correct()` → `save()` → `load()` → `predict()` round-trip. Verify that loaded predictions match original predictions (within tolerance).

**Research Reading:** "Software Testing for Machine Learning" — Google's ML testing guide. Sculley et al. (2015) "Hidden Technical Debt in Machine Learning Systems."

**Interview Question:** "A test passes 99 times out of 100 with a fixed seed. Is this acceptable? How would you debug the flaky 1%?"

**Reflection:** Week 4 complete. Can you now contribute to AdaptShot's test suite independently?

**Next-Day Prep:** What are the 10 core engineering principles of AdaptShot? Memorize them.

---

## Week 5: Systems Engineering & Architecture

### Day 21 — The 10 Principles as Design Constraints

**Learning Objectives:** Internalize every engineering principle and understand how each constrains design decisions.

**Theory:** Every principle from `.openproject.md` — CPU-First, Memory-Bound, Deterministic, Human-in-the-Loop, Transparent, Carbon-Aware, Offline-Capable, Torch-Optional, Correctness Over Convenience, Explicit Over Implicit. For each: what design decision does it enforce? What would break if we violated it?

**Mathematics:** None — architecture day.

**Algorithms:** None — architecture day.

**Software Engineering:** The frozen dataclass pattern for immutability. Lazy imports for optional dependencies. The `Literal` type for constrained config fields. Explicit error messages that tell the user what to do.

**AdaptShot Deep Dive:** Read `.openproject.md` again — now with deep understanding of every module. For each principle, identify 3 places in the codebase where it's enforced.

**Code Reading:**
- `.openproject.md` (all 621 lines)

**Coding Exercise:** For each of the 10 principles, write a hypothetical code review comment showing a violation and explaining how to fix it.

**Project Challenge:** Audit `FewShotLearner.predict()` against all 10 principles. Document any violations or near-violations. Propose fixes for each.

**Research Reading:** Re-read the original AdaptShot paper analysis in `research/AdaptShot_PaperAnalysis.md.pdf`.

**Interview Question:** "A contributor proposes adding a GPU-only Vision Transformer backbone. How do you evaluate this proposal against the 10 principles?"

**Reflection:** Which principle is hardest to satisfy? Which is most important to AdaptShot's mission?

**Next-Day Prep:** Review design patterns — Singleton, Strategy, Factory, Observer, Decorator. Which does AdaptShot use?

---

### Day 22 — Design Patterns in AdaptShot

**Learning Objectives:** Identify and evaluate the design patterns used throughout the codebase.

**Theory:**
- **Strategy Pattern**: `CalibrationEngine` with method="temperature"|"scaling_binning"|"conformal"|"none" — interchangeable calibration strategies
- **Factory Pattern**: `BackboneRegistry` — lazy factory functions for backbone creation
- **Observer Pattern**: `FeedbackRouter` — routes correction events to calibration, fine-tuning, buffer subsystems
- **Template Method**: `extract_embedding` — common preprocessing + backbone-specific forward pass
- **Facade Pattern**: `FewShotLearner` — simplifies the complex subsystem interactions behind a clean API
- **Singleton (via module-level cache)**: `_DEFAULT_CACHE`, `_build_backbone` with `@lru_cache`
- **Dependency Injection**: `FeedbackRouter` receives `calibrator` and `finetune_fn` as constructor arguments

**Mathematics:** None.

**Algorithms:** None.

**Software Engineering:**
- SOLID principles applied to AdaptShot:
  - **S**ingle Responsibility: Each core module does one thing
  - **O**pen/Closed: Calibration methods are extendable without modifying CalibrationEngine
  - **L**iskov Substitution: BackboneRegistry entries are interchangeable
  - **I**nterface Segregation: PredictionResult has only prediction-relevant fields
  - **D**ependency Inversion: FewShotLearner depends on abstractions (config), not concretions

**AdaptShot Deep Dive:** For each design pattern, find the exact lines of code that implement it. Evaluate: is this pattern appropriate here? Would another pattern be better?

**Code Reading:**
- `src/adaptshot/core/calibration.py` — Strategy Pattern
- `src/adaptshot/core/extractor.py` — Factory Pattern
- `src/adaptshot/training/feedback_router.py` — Observer Pattern

**Coding Exercise:** Refactor `CalibrationEngine.calibrate()` to use a dictionary dispatch instead of if/elif chains. Which is more maintainable? Evaluate tradeoffs.

**Project Challenge:** Add a pluggable similarity engine using the Strategy pattern. Create a `SimilarityEngine` base class and subclass implementations (NumPy, FAISS, custom). Wire it into `FewShotLearner`. Write tests.

**Research Reading:** Gamma et al. (1994) "Design Patterns: Elements of Reusable Object-Oriented Software" — read the Strategy and Observer chapters.

**Interview Question:** "AdaptShot uses dependency injection for FeedbackRouter. Why not just have FeedbackRouter import CalibrationEngine directly? What does DI buy us?"

**Reflection:** Which design pattern is most overused in software engineering? Which is most underused?

**Next-Day Prep:** Review error handling patterns. How does AdaptShot's exception hierarchy work?

---

### Day 23 — Error Handling and Defensive Programming

**Learning Objectives:** Master the error handling philosophy of AdaptShot.

**Theory:** Defensive programming — validate inputs early, fail fast, provide actionable error messages. The exception hierarchy: `AdaptShotError` → `InvalidImageError`, `ConfigValidationError`, `CalibrationNotReadyError`, `BufferCapacityError`. Graceful degradation: non-critical failures (buffer pruning, explanation) should not crash the prediction pipeline.

**Mathematics:** None.

**Algorithms:** None.

**Software Engineering:**
- Exception hierarchy design — flat vs deep, specific vs generic
- When to raise vs when to warn vs when to log
- The anti-pattern of bare `except:`
- `contextlib.suppress()` for expected failures
- Input validation as the first line of defense
- Atomic writes as crash protection

**AdaptShot Deep Dive:** Read every `raise` statement in `FewShotLearner`. Understand what each one guards against. Read `utils/exceptions.py`.

**Code Reading:**
- `src/adaptshot/utils/exceptions.py`
- Every `raise` in `src/adaptshot/core/learner.py`
- The `_validate_*` methods in `learner.py`

**Coding Exercise:** Write a function that demonstrates 5 different error handling anti-patterns (bare except, swallowing exceptions, generic messages, etc.), then fix them.

**Project Challenge:** Audit `FewShotLearner` for error handling gaps. Find at least 3 places where an error could occur but isn't caught or reported. Add appropriate error handling.

**Research Reading:** "Effective Java" Chapter 9 (Exceptions) — the principles apply to Python too.

**Interview Question:** "predict() catches no exceptions internally except calibration. Why? Is this a bug or by design?"

**Reflection:** How would you design error handling for a medical diagnosis system (where a missed error costs lives) vs a photo tagging app (where errors are annoying but not critical)?

**Next-Day Prep:** Review Python packaging. How does `pyproject.toml` define the project?

---

### Day 24 — Python Packaging and Dependency Management

**Learning Objectives:** Master Python packaging with PEP 621, optional dependencies, and lazy imports.

**Theory:** The Python packaging ecosystem: setuptools, wheel, pyproject.toml. PEP 621 — declarative project metadata. Optional dependencies with extras: `[torch]`, `[faiss]`, `[ui]`, `[gui]`, `[dev]`. The `[all]` meta-extra. CLI entry points via `[project.scripts]`.

**Mathematics:** None.

**Algorithms:** None.

**Software Engineering:**
- `pyproject.toml` structure: `[build-system]`, `[project]`, `[tool.*]`
- `setuptools.packages.find` with `where=["src"]`
- `package-data` for non-code files (ONNX models)
- The `_get_*()` lazy import pattern — why it's essential for torch-optional
- `TYPE_CHECKING` guard for type-only imports
- Development installs: `pip install -e ".[dev]"`

**AdaptShot Deep Dive:** Read `pyproject.toml` line by line. Understand every section. Trace how a `pip install adaptshot[torch]` resolves dependencies.

**Code Reading:**
- `pyproject.toml`
- `src/adaptshot/core/extractor.py` lazy import helpers
- `src/adaptshot/core/learner.py` lazy import helpers

**Coding Exercise:** Create a minimal Python package with the same structure as AdaptShot: `src/` layout, pyproject.toml, optional dependency with lazy import. Verify it installs and imports correctly.

**Project Challenge:** The `[gui]` extra in pyproject.toml includes `onnxscript>=0.3.0`. Verify that this dependency is actually needed. If not, remove it and update the docs.

**Research Reading:** Python Packaging User Guide — "Packaging Python Projects" tutorial. PEP 621 specification.

**Interview Question:** "A user runs `pip install adaptshot` and gets `ModuleNotFoundError: No module named 'torch'` when calling `predict()`. What went wrong and how would you fix it?"

**Reflection:** What are the benefits and drawbacks of AdaptShot's two-dependency core (numpy + Pillow)? Could we reduce to one? Should we?

**Next-Day Prep:** Review memory profiling. How is the <250MB budget enforced?

---

### Day 25 — Memory Profiling and Optimization

**Learning Objectives:** Master memory profiling techniques and enforce the <250MB RAM budget.

**Theory:** Python memory measurement: RSS (Resident Set Size), VMS (Virtual Memory Size), heap size. `tracemalloc` for allocation tracing. `psutil` for process memory. NumPy array memory: `arr.nbytes` for exact, `sys.getsizeof(arr)` for object overhead. The `float32` vs `float64` memory doubling.

**Mathematics:**
- Array memory: \( \text{bytes} = N \times D \times 4 \) (for float32), \( N \times D \times 8 \) (for float64)
- 1000 support examples × 512-dim × float32 = 2,048,000 bytes ≈ 2MB
- 1000 support examples × 512-dim × float64 = 4,096,000 bytes ≈ 4MB
- Why this matters: support embeddings dominate memory, not the backbone

**Algorithms:** Memory-mapped I/O (`np.memmap`) for large arrays. Streaming processing to avoid materializing large arrays.

**Software Engineering:** The `MemoryTracker` class in `utils/profiling.py`. Context managers for memory measurement. The `estimate_model_memory_mb` function. Benchmark validation of memory claims.

**AdaptShot Deep Dive:** Read `utils/profiling.py` and `benchmarks/energy_profile.py`. Understand how memory is measured.

**Code Reading:**
- `src/adaptshot/utils/profiling.py`
- `benchmarks/energy_profile.py`

**Coding Exercise:** Write a memory profiler that measures peak memory usage of a function using tracemalloc. Test it on `load_support_images` with varying support set sizes (10, 50, 100, 200 images).

**Project Challenge:** Profile `FewShotLearner.predict()` for memory leaks. Call predict 1000 times in a loop. Does memory grow unbounded? If so, find the leak and fix it. Write a regression test.

**Research Reading:** "Memory Management in Python" (Real Python). The NumPy memory model documentation.

**Interview Question:** "A user reports that loading 500 support images uses 400MB RAM instead of the expected ~10MB. List 5 possible causes and how you'd diagnose each."

**Reflection:** Week 5 complete. Can you now optimize any AdaptShot function for memory?

**Next-Day Prep:** Review the benchmark suite. What benchmarks exist, and what do they measure?

---

## Week 6: Performance, Benchmarks, and Production

### Day 26 — Benchmarking and Reproducibility

**Learning Objectives:** Master the benchmark suite and understand all performance metrics.

**Theory:** Why benchmarks matter: they validate claims, catch regressions, and inform design. The benchmark dimensions: accuracy, calibration (ECE), latency, memory, energy. Reproducibility requirements: fixed seed, documented hardware, identical software versions. The smoke test as CI gate.

**Mathematics:**
- Expected Calibration Error (ECE): \( \text{ECE} = \sum_{b=1}^B \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)| \)
- Debiased ECE: accounts for finite-sample bias in bin-wise accuracy estimates
- AUROC: Area Under Receiver Operating Characteristic — OOD detection metric
- FPR95: False Positive Rate at 95% True Positive Rate — stricter OOD metric

**Algorithms:** Benchmark harness design — parameterized test runs, result aggregation, assertion-based validation.

**Software Engineering:**
- `argparse` for CLI flags (`--smoke-test`, `--full-benchmark`, `--profile-memory`, `--seed`)
- JSON result serialization for comparison across runs
- The benchmark module structure: `run_benchmark.py`, `day2_integration.py`, `energy_profile.py`

**AdaptShot Deep Dive:** Read all benchmark files. Run the smoke test.

**Code Reading:**
- `benchmarks/run_benchmark.py`
- `benchmarks/day2_integration.py`
- `benchmarks/energy_profile.py`

**Coding Exercise:** Write a benchmark that measures: (1) latency of predict() for 100 queries, (2) P50, P95, P99 latencies, (3) memory before and after. Output as JSON.

**Project Challenge:** Add a `--hardware-matrix` flag to `run_benchmark.py` that runs all benchmarks and outputs a machine-readable JSON with all metrics (accuracy, ECE, latency P50/P95/P99, peak RAM, Joules/inference). Write the output to `results/hardware_matrix.json`.

**Research Reading:** "How to Benchmark ML Systems" — read industry best practices from MLCommons (MLPerf).

**Interview Question:** "A benchmark shows 92.3% accuracy with seed=42. A reviewer asks: 'What is the 95% confidence interval?' How do you compute it?"

**Reflection:** What benchmark is most important for AdaptShot's mission? What metric would convince a skeptic?

**Next-Day Prep:** Review type hints and static analysis. What does `mypy --strict` check?

---

### Day 27 — Type Safety and Static Analysis

**Learning Objectives:** Master Python type hints and static analysis with mypy.

**Theory:** Type systems — static vs dynamic, nominal vs structural. Python's gradual typing. Why type hints matter for large codebases: catch bugs before runtime, enable IDE autocompletion, serve as documentation. The `mypy --strict` flag enables all checks.

**Mathematics:** None.

**Algorithms:** None.

**Software Engineering:**
- `from __future__ import annotations` — enables forward references
- `TYPE_CHECKING` guard for circular imports
- `Optional[X]`, `Union[X, Y]`, `Literal["a", "b"]`, `TypeVar`
- `cast()` for when you know more than the type checker
- Protocol classes (PEP 544) for structural subtyping
- `Any` as escape hatch — when to use and when not to
- Generics: `List[str]`, `Dict[str, float]`, `Callable[[int], str]`

**AdaptShot Deep Dive:** Look at every type annotation in `learner.py`. Understand why `_get_torch()` returns `Any`. Why is `_sim_embeddings` typed as `List[np.ndarray]` instead of `np.ndarray`?

**Code Reading:**
- Check type hints in any AdaptShot file

**Coding Exercise:** Take a function you wrote in Week 1-2 and add complete type annotations. Run `mypy --strict` and fix every error.

**Project Challenge:** Run `mypy src/adaptshot --strict` and fix any type errors. If there are none, find 3 places where type annotations could be more precise and improve them.

**Research Reading:** PEP 484 (Type Hints), PEP 526 (Variable Annotations), PEP 544 (Protocols).

**Interview Question:** "Why does `FewShotLearner._get_torch()` return `Any` instead of `torch`? What would break if we used the real type?"

**Reflection:** Do type hints make code harder to read? When would you omit them?

**Next-Day Prep:** Review linting and code quality. What does `ruff check` enforce?

---

### Day 28 — Code Quality and Linting

**Learning Objectives:** Master code quality enforcement with ruff.

**Theory:** Why linting matters: consistent style reduces cognitive load, catches common bugs (unused imports, undefined names), enforces best practices. Ruff as a fast Python linter (10-100x faster than flake8). The rule categories: pyflakes (bugs), pycodestyle (style), isort (import ordering), pydocstyle (docstrings).

**Mathematics:** None.

**Algorithms:** None.

**Software Engineering:**
- `pyproject.toml` ruff configuration: `target-version`, `line-length`, `select`, `ignore`
- Common ruff rules and what they catch
- `# noqa` comments — when justified and when lazy
- Auto-fixing: `ruff check --fix`
- Pre-commit hooks integration

**AdaptShot Deep Dive:** Review the ruff configuration in `pyproject.toml`. Run `ruff check src/`. Fix any issues.

**Code Reading:**
- `pyproject.toml` lines 56-58

**Coding Exercise:** Take a messy Python file and run ruff on it. Fix every issue. Observe how the code quality improves.

**Project Challenge:** Add docstring enforcement rules to ruff config for public API modules. Run on `src/adaptshot/` and fix all missing or malformed docstrings. Ensure Google-style compliance.

**Research Reading:** "The Ruff Formatter" documentation. The PEP 8 style guide.

**Interview Question:** "A contributor uses `import torch` at the module level in a new file. Ruff doesn't catch this. Why not, and how would you enforce the lazy import convention?"

**Reflection:** Is strict linting worth the overhead? When have linting rules saved you from a real bug?

**Next-Day Prep:** Review API design. What makes a good API?

---

### Day 29 — API Design Philosophy

**Learning Objectives:** Master the scikit-learn-inspired API design of AdaptShot.

**Theory:** API design principles: small surface area, consistency over cleverness, sensible defaults, return dataclasses not dicts, immutable config. The scikit-learn convention: `fit()`/`predict()`/`predict_proba()`. AdaptShot's convention: `load_support_images()` (load data + fit) → `predict()` → `correct()`.

**Mathematics:** None.

**Algorithms:** None.

**Software Engineering:**
- Public vs private API: `__all__` in `__init__.py`
- Deprecation cycle: `DeprecationWarning` → support old + new → remove old
- Breaking change policy: one minor version of deprecation before removal
- Experimental features: `ExperimentalWarning`, disabled by default, gated by config flags
- Semantic versioning: MAJOR.MINOR.PATCH

**AdaptShot Deep Dive:** Read `src/adaptshot/__init__.py` — this IS the public API contract. For each export, verify it has: Google docstring, type hints, tests, documentation.

**Code Reading:**
- `src/adaptshot/__init__.py`
- `src/adaptshot/config/settings.py` (the frozen dataclass pattern)

**Coding Exercise:** Design a public API for a new Feature: "Model Ensemble" — combine predictions from multiple backbones. Write the `__init__.py` exports, the main class signature, and a usage example.

**Project Challenge:** The `ContrastivePrototypeLearner` API has some rough edges — `refine_prototypes` must be called before `project_query`. Design a cleaner API that makes misuse impossible (e.g., constructor takes support data, projection happens automatically). Write a migration guide.

**Research Reading:** "How to Design a Good API and Why it Matters" — Joshua Bloch's classic talk. Read the transcript.

**Interview Question:** "AdaptShot's predict() returns a PredictionResult dataclass. A user wants just the class label. How would you handle this without breaking existing users?"

**Reflection:** What is the best-designed API you've ever used? What makes it great?

**Next-Day Prep:** Review the full AdaptShot documentation. What's documented and what's not?

---

### Day 30 — Documentation and Technical Writing

**Learning Objectives:** Master technical documentation — docstrings, tutorials, API references.

**Theory:** Documentation as a first-class deliverable. Three audiences: new users (tutorials, quickstart), practitioners (API reference, guides), contributors (architecture docs, code comments). Google-style docstrings: `Args:`, `Returns:`, `Raises:`, `Examples:`. MkDocs + Material + mkdocstrings for auto-generated docs.

**Mathematics:** None.

**Algorithms:** None.

**Software Engineering:**
- Docstring conventions: one-line summary, blank line, detailed description, sections
- Code examples in docstrings — must be runnable
- Never document unimplemented features ("coming soon" is forbidden)
- CHANGELOG.md maintenance (Keep a Changelog format)
- Migration guides for breaking changes

**AdaptShot Deep Dive:** Read 5 random docstrings. Evaluate: complete? Accurate? Runnable? Read `docs/api/` and `docs/tutorials/`.

**Code Reading:**
- Random sample of docstrings in any module
- `CHANGELOG.md`
- `docs/tutorials/01_getting_started.md`

**Coding Exercise:** Write a complete Google-style docstring for a function of your choice. Include Args, Returns, Raises, and a runnable Example. Verify with `doctest` if possible.

**Project Challenge:** The `ConformalEngine` module docstring is good but could be more thorough. Rewrite it to include: mathematical background (conformal p-value formula), usage example, known limitations, references. Update the class docstrings for `predict_set` and `predict_set_class_conditional`.

**Research Reading:** "What Nobody Tells You About Documentation" — Divio's documentation system tutorial.

**Interview Question:** "A new user reports: 'The docs say predict() returns a PredictionResult, but my code gets an AttributeError when accessing .prediction.' What documentation bug could cause this?"

**Reflection:** Week 6 complete. Can you now maintain AdaptShot's documentation independently?

**Next-Day Prep:** Review research papers. How do you read an ML paper effectively?

---

## Week 7: Research Integration

### Day 31 — Reading ML Research Papers

**Learning Objectives:** Develop the skill of reading, understanding, and critically evaluating ML research papers.

**Theory:** The three-pass method (Keshav): (1) quick skim to get category and context (5-10 min), (2) careful read of figures and results (1 hour), (3) virtually re-implement the paper (4-5 hours). Paper anatomy: abstract, introduction, related work, method, experiments, conclusion. How to identify the core contribution vs incremental improvements.

**Mathematics:** None — meta-skill day.

**Algorithms:** None — meta-skill day.

**Software Engineering:** None — meta-skill day.

**AdaptShot Deep Dive:** Read `research/AdaptShot_PaperAnalysis.md.pdf` — the analysis of the original AdaptShot paper. What claims does it make? What evidence supports them?

**Code Reading:**
- `research/AdaptShot_PaperAnalysis.md.pdf`
- `research/RESEARCH_AGENDA_2026_2028.md`

**Coding Exercise:** Take the Prototypical Networks paper (Snell et al., 2017). Apply the three-pass method. Write a 1-page summary: what's the core idea? What experiments validate it? What are the limitations? What would you improve?

**Project Challenge:** For each algorithm in AdaptShot, find its source paper and verify that the implementation matches the paper's description. Document any discrepancies. For each discrepancy, determine: intentional adaptation or bug?

**Research Reading:** Read one of the papers in `research/` that you haven't read yet.

**Interview Question:** "A paper claims 95% accuracy on miniImageNet 5-way 1-shot. AdaptShot gets 62%. Is AdaptShot worse? List 5 possible explanations for the gap."

**Reflection:** What paper was hardest to understand? What made it hard? How would you make it more accessible?

**Next-Day Prep:** Choose one algorithm from AdapShot to reimplement from scratch.

---

### Day 32 — Implement Temperature Scaling from Scratch

**Learning Objectives:** Reimplement `CalibrationEngine` from scratch, then compare with the existing implementation.

**Theory:** Review temperature scaling: grid search over T ∈ [0.5, 3.0] to minimize cross-entropy loss on calibration data. The difference between ECE optimization (non-differentiable) and NLL optimization (differentiable).

**Mathematics:**
- Logit transform: \( l = \ln(p / (1-p)) \)
- Temperature scaling: \( p_{\text{cal}} = \sigma(l / T) \)
- NLL loss: \( L(T) = -\frac{1}{N} \sum_i [y_i \log(p_i) + (1-y_i) \log(1-p_i)] \)

**Algorithms:** Grid search, binary search for refinement, sliding window for online updates.

**Software Engineering:** Class design — what state does the calibrator need? What methods should be public vs private?

**AdaptShot Deep Dive:** Close the calibration.py file. Reimplement temperature scaling + ECE computation + sliding window + grid search from memory. Then compare with the original.

**Code Reading:** None — implementation day.

**Coding Exercise:** Implement a `TemperatureScaler` class with: `fit(confidences, correct_labels)` → grid search T, `transform(confidences)` → apply T, `fit_transform` → convenience method. Write tests.

**Project Challenge:** Extend your implementation with Platt scaling (logistic regression) as an alternative calibration method. Compare temperature scaling vs Platt scaling on synthetic overconfident predictions. Which works better? Why?

**Research Reading:** Platt (1999) "Probabilistic Outputs for Support Vector Machines" — the original calibration paper. Guo et al. (2017) — re-read the temperature scaling section.

**Interview Question:** "Why does temperature scaling use NLL (cross-entropy) as the optimization objective instead of ECE directly?"

**Reflection:** What was hardest about implementing from scratch? What did you learn that reading alone wouldn't teach you?

**Next-Day Prep:** Prepare to implement InfoNCE contrastive learning from scratch.

---

### Day 33 — Implement Contrastive Prototype Learning from Scratch

**Learning Objectives:** Reimplement the contrastive prototype learner from first principles.

**Theory:** Full review of InfoNCE, projection head training, prototype refinement. The forward and backward passes through a 2-layer MLP.

**Mathematics:**
- Full InfoNCE gradient derivation (backprop through projection head)
- SGD with momentum
- Cross-entropy prototype refinement

**Algorithms:** Mini-batch SGD. Matrix operations for batched computation. L2 normalization for cosine similarity.

**Software Engineering:** Modular design — separate projection head training from prototype refinement. Test each component independently.

**AdaptShot Deep Dive:** Close contrastive.py. Reimplement from memory.

**Code Reading:** None — implementation day.

**Coding Exercise:** Implement `ContrastivePrototypeLearner` from scratch with: (1) 2-layer MLP projection head with He init, (2) InfoNCE loss computation and gradient, (3) backpropagation through the head via SGD+momentum, (4) prototype refinement via cross-entropy gradient descent. Test on synthetic 2D data where you can visualize the prototypes.

**Project Challenge:** Add a `visualize_projection_space` method that projects both embeddings and prototypes to 2D using PCA. Show before/after contrastive refinement. Verify visually that classes are better separated after training.

**Research Reading:** Khosla et al. (2020) "Supervised Contrastive Learning" — extends SimCLR to use labels.

**Interview Question:** "The contrastive learner trains the projection head THEN refines prototypes. Why two stages? What would happen if we trained them jointly?"

**Reflection:** Compare your implementation with AdaptShot's. Where did you make different design choices? Which is better?

**Next-Day Prep:** Prepare to implement conformal prediction from scratch.

---

### Day 34 — Implement Conformal Prediction from Scratch

**Learning Objectives:** Reimplement split-conformal prediction from first principles.

**Theory:** Exchangeability, nonconformity scores, quantile computation with finite-sample correction, prediction set construction.

**Mathematics:**
- Conformal quantile: \( \hat{q} = Q_{1-\alpha}(s_1, ..., s_n) \) with correction \( \lceil (n+1)(1-\alpha) \rceil / n \)
- Coverage guarantee: \( P(Y_{n+1} \in \hat{C}(X_{n+1})) \geq 1 - \alpha \) under exchangeability
- Softmax nonconformity: \( s = 1 - \text{softmax}(y_{\text{true}} | \text{distances}) \)

**Algorithms:** Quantile computation, leave-one-out calibration, prediction set construction.

**Software Engineering:** Separating calibration from inference. The rolling calibration buffer. Class-conditional vs global conformal.

**AdaptShot Deep Dive:** Close conformal.py. Reimplement from memory.

**Code Reading:** None — implementation day.

**Coding Exercise:** Implement split-conformal prediction for a multi-class problem: (1) compute nonconformity scores on calibration set, (2) compute quantile threshold, (3) build prediction set for test example. Verify that empirical coverage ≥ 1-α on held-out test data.

**Project Challenge:** Implement cross-conformal prediction (k-fold). Compare with split-conformal: is coverage more stable? Are sets larger or smaller? Write a report.

**Research Reading:** Barber et al. (2021) "Predictive Inference with the Jackknife+" — an improvement over standard cross-conformal.

**Interview Question:** "A user reports: 'My conformal prediction sets are always empty for α=0.05.' What could be wrong? How would you debug this?"

**Reflection:** Do you now feel confident implementing conformal prediction in any ML system?

**Next-Day Prep:** Prepare for the Capstone Week. You'll redesign subsystems independently.

---

### Day 35 — Comparative Analysis: AdaptShot vs the Field

**Learning Objectives:** Critically evaluate AdaptShot against the broader ML ecosystem.

**Theory:** The few-shot learning landscape: Prototypical Networks, Matching Networks, MAML, MetaOptNet, FEAT, DeepEMD. The calibration landscape: Temperature Scaling, Isotonic Regression, Beta Calibration, Platt Scaling. The OOD landscape: Mahalanobis, Energy-based, GradNorm, ViM.

**Mathematics:** None — comparative analysis day.

**Algorithms:** None — comparative analysis day.

**Software Engineering:** How do other libraries structure their code? Compare AdaptShot with: scikit-learn, PyTorch Lightning, HuggingFace Transformers, MONAI. What patterns do they share? What does AdaptShot do differently?

**AdaptShot Deep Dive:** For each subsystem, identify the closest analog in another library. Evaluate: which design is better? Why?

**Code Reading:**
- Skim scikit-learn's `CalibratedClassifierCV` source
- Skim PyTorch Lightning's `LightningModule` source
- Skim HuggingFace's `pipeline` source

**Coding Exercise:** Write a comparison table: AdaptShot vs scikit-learn vs HuggingFace for 10 dimensions (ease of use, CPU support, memory efficiency, calibration quality, few-shot accuracy, documentation quality, test coverage, type safety, extensibility, carbon footprint). Score each 1-5.

**Project Challenge:** Write an "AdaptShot vs X" guide for the AdaptShot docs. Compare with one competitor on: API design, performance, memory, calibration, HITL support. Be honest about AdaptShot's weaknesses.

**Research Reading:** Browse the latest NeurIPS/ICML/CVPR proceedings. Find 3 papers that could improve AdaptShot.

**Interview Question:** "If you could steal one feature from another ML library and add it to AdaptShot, what would it be and why?"

**Reflection:** Week 7 complete. What is AdaptShot's biggest competitive advantage? Biggest weakness?

**Next-Day Prep:** Prepare for the Capstone. Choose a subsystem to redesign.

---

## Week 8: Capstone — Redesign AdaptShot

### Day 36 — Capstone 1: Redesign the Calibration Engine

**Learning Objectives:** Independently redesign a core subsystem with justification for every decision.

**Theory:** The calibration engine has three methods (temperature, scaling_binning, conformal) with shared state. This creates complexity — adding a new method requires touching the core calibrate() logic.

**Mathematics:** Review all calibration methods and suggest improvements.

**Algorithms:** Design a pluggable calibration architecture using the Strategy pattern.

**Software Engineering:** Write a design document (ADR format): Context, Decision, Alternatives, Consequences.

**AdaptShot Deep Dive:** The entire `calibration.py` module — reimagine it.

**Code Reading:** `src/adaptshot/core/calibration.py` — full review.

**Coding Exercise:** Implement your redesigned calibration engine. Must include: (1) clear interface for adding new calibration methods, (2) no if/elif chains in calibrate(), (3) backward compatibility with existing CalibrationEngine API.

**Project Challenge:** Implement a Bayesian Binning into Quantiles (BBQ) calibration method as a new backend. Compare calibration quality with temperature scaling on few-shot predictions.

**Research Reading:** Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using Bayesian Binning" — the BBQ paper.

**Interview Question:** "Defend your calibration engine redesign. Why is it better? What does it make harder?"

**Reflection:** What was the hardest design trade-off? What would you do differently next time?

---

### Day 37 — Capstone 2: Redesign the Similarity Engine

**Learning Objectives:** Design a pluggable similarity search backend.

**Theory:** The current similarity.py supports NumPy and FAISS via if/else branches. A properly pluggable design would allow ANNOY, HNSWlib, ScaNN, and custom backends to be added without modifying core logic.

**Mathematics:** Compare approximate nearest neighbor algorithms: IVF, HNSW, LSH, PQ — accuracy vs speed vs memory tradeoffs.

**Algorithms:** Design an abstract `SimilarityBackend` interface with concrete implementations for each algorithm.

**Software Engineering:** Protocol class, abstract base class, or duck typing? Dependency injection — `FewShotLearner` receives a `SimilarityBackend` instead of importing similarity.py directly.

**AdaptShot Deep Dive:** The entire `similarity.py` module — reimagine it.

**Code Reading:** `src/adaptshot/core/similarity.py` — full review.

**Coding Exercise:** Implement a `SimilarityBackend` protocol + `NumPyBackend` + `FAISSBackend` implementations. Wire them into `FewShotLearner` via config.

**Project Challenge:** Add an HNSW backend using `hnswlib` library. Benchmark against NumPy and FAISS on 100, 1000, 10000 support examples. Write results.

**Research Reading:** Malkov & Yashunin (2018) "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" — the HNSW paper.

**Interview Question:** "Your pluggable similarity engine adds ~200 lines of abstraction overhead. A reviewer says: 'This is over-engineering — we only need two backends.' How do you respond?"

**Reflection:** When is abstraction worth the cost? When is it premature optimization?

---

### Day 38 — Capstone 3: Redesign the Uncertainty System

**Learning Objectives:** Redesign the multi-signal uncertainty system for extensibility.

**Theory:** The current UncertaintyQuantifier computes three signals in one class. A cleaner design would treat each signal as an independent "detector" that produces a score + metadata.

**Mathematics:** Design a composite uncertainty fusion method that supports learned weights, not just static weights.

**Algorithms:** Weight learning via logistic regression on labeled OOD/in-distribution data. Dynamic weight adjustment based on data regime (few-shot vs many-shot).

**Software Engineering:** Observer pattern — each detector reports its score, a fusion layer combines them. The `UncertaintyReport` becomes the aggregation point.

**AdaptShot Deep Dive:** The entire `uncertainty.py` module — reimagine it.

**Code Reading:** `src/adaptshot/core/uncertainty.py` — full review.

**Coding Exercise:** Implement `EpistemicDetector`, `AleatoricDetector`, `DistributionalDetector` as independent classes. Create `UncertaintyFusion` that combines them with configurable weights. Implement learned weight calibration.

**Project Challenge:** Add a `GradientDetector` that computes uncertainty from the gradient of loss w.r.t. input (requires torch). Compare its OOD detection performance with Mahalanobis.

**Research Reading:** Liang et al. (2018) "Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks" — ODIN method.

**Interview Question:** "A new user says: 'I don't understand which uncertainty mode to use.' How would you simplify the API?"

**Reflection:** Is multi-signal uncertainty worth the complexity? Could a single well-tuned signal match the ensemble?

---

### Day 39 — Capstone 4: Full System Design Review

**Learning Objectives:** Review your capstone redesigns. Defend them as if presenting to the AdaptShot Review Committee.

**Theory:** The full AdaptShot system — now through the lens of someone who can rebuild it.

**Mathematics:** None — review day.

**Algorithms:** None — review day.

**Software Engineering:** Prepare a design review presentation (in writing). For each subsystem you redesigned: (1) what was wrong with the original? (2) what does your redesign improve? (3) what tradeoffs did you make? (4) what would you do differently with more time?

**AdaptShot Deep Dive:** The entire codebase — one final reading pass.

**Code Reading:** Any module you still feel uncertain about.

**Coding Exercise:** Write a comprehensive design review document covering all three capstone redesigns. Include: problem statement, proposed solution, alternatives considered, migration path, risk assessment.

**Project Challenge:** Identify 3 concrete improvements you can make to AdaptShot based on your capstone work. Prioritize them by impact. Create implementation plans for each.

**Research Reading:** Re-read `.openproject.md` — with 40 days of deep understanding behind you.

**Interview Question:** "You've been given 6 months and a team of 3 to build AdaptShot v1.0. What do you prioritize? What do you leave for later? Justify every decision."

**Reflection:** Day 39. One day left. What do you still not understand? What do you want to learn next?

---

### Day 40 — Final Examination: Build AdaptShot from Scratch

**Learning Objectives:** Demonstrate mastery by sketching the complete AdaptShot architecture from memory, defending every decision.

**Theory:** Every theory, equation, and algorithm from Days 1-39 — synthesized.

**Mathematics:** Every equation in AdaptShot — derived from first principles.

**Algorithms:** Every algorithm — implemented or outlined.

**Software Engineering:** The full architecture — modules, classes, interfaces, data flow.

**AdaptShot Deep Dive:** The entire library — one final mastery demonstration.

**Code Reading:** None — you ARE the codebase now.

**Coding Exercise:** THE FINAL EXAM. Without looking at the source code:
1. Write the module structure (directories and files)
2. Write the `AdaptShotConfig` dataclass with all 26 fields and their defaults
3. Write the `PredictionResult` dataclass with all fields
4. Sketch the `FewShotLearner` class with all public methods and their signatures
5. Trace the `predict()` pipeline — every function call, every data transformation
6. Trace the `correct()` pipeline — every subsystem interaction
7. Write `cosine_similarity_numpy` from memory
8. Write `compute_ece` from memory
9. Write `_compute_infonce_loss` from memory
10. Write the conformal quantile formula from memory

Then open the source code and compare. Grade yourself.

**Project Challenge:** Based on everything you've learned, write AdaptShot v0.3.0 ROADMAP. What should be added, removed, refactored? Prioritize by impact on AdaptShot's mission.

**Research Reading:** The research agenda you want to pursue after Day 40.

**Interview Question — THE FINAL:** "You built AdaptShot. Tell me why it exists, why it's designed the way it is, what you're proudest of, and what you would do differently if you started over."

**Reflection:** 40 days. You started as someone who built AdaptShot with AI assistance. You end as the engineer who understands every line, every equation, every trade-off, and every decision. What will you build next?

---

## Success Criteria

After 40 days, you should be able to:

1. **Explain every module** — walk through the codebase without looking at source files
2. **Defend every architectural decision** — justify why each subsystem exists and why it's designed that way
3. **Improve the library independently** — identify bugs, add features, refactor code without AI guidance
4. **Design better algorithms** — propose and implement improvements to existing algorithms
5. **Answer technical interview questions** — on ML systems, few-shot learning, calibration, uncertainty
6. **Explain the mathematics** — derive every equation in the codebase from first principles
7. **Implement features from research papers** — read a paper and translate it into AdaptShot code
8. **Review pull requests** — critically evaluate code changes for correctness, performance, and design
9. **Write production-quality code** — that passes ruff, mypy, pytest, and the smoke test
10. **Publish research based on this library** — design experiments, run benchmarks, write papers
