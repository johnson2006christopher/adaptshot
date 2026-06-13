# Tutorial 15: Advanced Uncertainty Quantification

> **v0.2.0** | Multi-signal uncertainty estimation and OOD detection

---

## Prerequisites

- AdaptShot v0.2.0+ installed
- Completed [Tutorial 1: Getting Started](01_getting_started.md)

---

## Understanding Uncertainty

AdaptShot decomposes prediction uncertainty into three complementary signals:

| Signal | Type | What it measures |
|--------|------|-----------------|
| **Epistemic** | Model uncertainty | "Has the model seen similar data?" |
| **Aleatoric** | Data uncertainty | "Are the class boundaries ambiguous?" |
| **Distributional** | OOD uncertainty | "Is this input from a known distribution?" |

---

## Step 1: Basic Uncertainty Quantification

```python
import numpy as np
from adaptshot import UncertaintyQuantifier

# Initialize with 95th percentile for OOD detection
uq = UncertaintyQuantifier(ood_percentile=95.0)

# Fit class distributions from support embeddings
embeddings = np.random.randn(20, 64).astype(np.float32)
labels = np.array(["cat"] * 10 + ["dog"] * 10, dtype=object)
uq.fit_class_distributions(embeddings, labels)

# Quantify uncertainty for a query
query = np.random.randn(64).astype(np.float32)
report = uq.quantify(query, embeddings, labels)

print(f"Epistemic:    {report.epistemic:.3f}")
print(f"Aleatoric:    {report.aleatoric:.3f}")
print(f"Distributional: {report.distributional:.3f}")
print(f"Composite:    {report.composite:.3f}")
print(f"OOD:          {report.is_ood}")
```

---

## Step 2: Mahalanobis Distance for OOD Detection

The Mahalanobis distance accounts for class covariance structure:

```python
# Query at class "cat" center (in-distribution)
cat_mean = uq._class_means["cat"]
dist_to_cat = uq.mahalanobis_distance(cat_mean, "cat")
dist_to_dog = uq.mahalanobis_distance(cat_mean, "dog")

print(f"Distance to own class: {dist_to_cat:.3f}")   # Low
print(f"Distance to other class: {dist_to_dog:.3f}")  # Higher

# Far outlier (OOD)
outlier = np.ones(64, dtype=np.float32) * 50.0
is_ood, score = uq.is_ood(outlier)
print(f"Outlier is OOD: {is_ood}, score: {score:.3f}")
```

---

## Step 3: k-NN Entropy (Aleatoric Uncertainty)

Entropy over nearest neighbors reveals class boundary ambiguity:

```python
entropy, norm_entropy = uq.compute_knn_entropy(
    query_embedding=query,
    support_embeddings=embeddings,
    support_labels=labels,
)
print(f"Raw entropy: {entropy:.3f}")
print(f"Normalized entropy: {norm_entropy:.3f}")

# Interpretation:
# - entropy ≈ 0: query is clearly in one class region
# - entropy ≈ 1: query is on a class boundary
```

---

## Step 4: Using Uncertainty with FewShotLearner

The learner automatically computes uncertainty on every prediction:

```python
from adaptshot import FewShotLearner, AdaptShotConfig

config = AdaptShotConfig(
    device="cpu",
    uncertainty_mode="entropy",
    enable_ood_detection=True,
)

learner = FewShotLearner(config=config)
learner.load_support_images(
    ["cat_01.jpg", "cat_02.jpg", "dog_01.jpg"],
    ["cat", "cat", "dog"],
)

result = learner.predict("query.jpg")
report = result.uncertainty_report

print(f"Composite uncertainty: {report['composite']:.3f}")
print(f"Is OOD: {bool(report['is_ood'])}")

if report["composite"] > 0.5:
    print("⚠️ High uncertainty — consider requesting human feedback")
```

---

## Step 5: Interpreting Uncertainty Reports

```python
report = result.uncertainty_report

# Decision logic based on uncertainty decomposition
if report["is_ood"]:
    print("❌ Input is out-of-distribution. Do not trust prediction.")
elif report["composite"] > 0.3:
    print("⚠️ Moderate uncertainty. Confidence may be unreliable.")
elif report["entropy"] > 0.5:
    print("⚠️ High data uncertainty. Class boundaries may be ambiguous.")
else:
    print("✅ Low uncertainty. Prediction is reliable.")
```

---

## OOD Detection Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ood_percentile` | 95.0 | Higher = fewer false OOD flags |
| `min_ood_samples` | 10 | Minimum samples before OOD activates |
| `mahalanobis_regularization` | 1e-4 | Ridge term for singular covariance |

```python
# Looser OOD detection (fewer flags)
uq_lenient = UncertaintyQuantifier(ood_percentile=99.0)

# Stricter OOD detection (more flags)
uq_strict = UncertaintyQuantifier(ood_percentile=90.0)
```

---

## Next Steps

- [Algorithm Theory: Multi-Signal Uncertainty](../guides/algorithm-theory.md#9-multi-signal-uncertainty-v020)
- [Tutorial 16: Explainability & XAI](16_explainability.md)
- [Tutorial 18: End-to-End Production Workflow](18_end_to_end_workflow.md)
