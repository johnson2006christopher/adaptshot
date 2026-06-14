# Tutorial 18: End-to-End Production Workflow

> **v0.2.0** | Complete pipeline: support loading → inference → feedback → profiling → monitoring

---

## Prerequisites

- AdaptShot v0.2.0+ installed with `pip install "adaptshot[torch]"`
- A directory of labeled RGB images
- Familiarity with [Quickstart](../getting-started/quickstart.md)

---

## Step 1: Project Setup

```python
import json
import time
from pathlib import Path
from adaptshot import FewShotLearner, AdaptShotConfig

# Production configuration with all v0.2.0 features
config = AdaptShotConfig(
    device="cpu",
    backbone="mobilenet_v3_small",  # Lighter for production
    inference_mode="prototypical",
    calibration_method="temperature",
    conformity_alpha=0.05,           # 95% coverage
    uncertainty_mode="ensemble",
    explainability_enabled=True,
    enable_ood_detection=True,
    max_buffer_size=200,             # Larger buffer for production
    recalibrate_after_feedback=True,
)

learner = FewShotLearner(config=config)
```

---

## Step 2: Load Initial Support Set

```python
def load_support_from_directory(data_dir: str):
    """Load support images organized in class subdirectories."""
    image_paths = []
    labels = []
    
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.glob("*.jpg"):
                image_paths.append(str(img_path))
                labels.append(class_dir.name)
    
    print(f"Loading {len(image_paths)} images across {len(set(labels))} classes")
    learner.load_support_images(image_paths, labels)
    
    # Save initial state
    learner.save("production_checkpoint.json")
    return len(set(labels))

num_classes = load_support_from_directory("./data/support_set/")
print(f"{num_classes} classes loaded")
```

---

## Step 3: Inference with Full Diagnostics

```python
def run_inference(image_path: str):
    """Run inference and collect all diagnostics."""
    result = learner.predict(image_path)
    
    # Basic prediction
    prediction = {
        "class": result.prediction,
        "confidence": result.calibrated_confidence,
        "raw_confidence": result.raw_confidence,
        "act_action": result.act_action,
    }
    
    # Conformal prediction set
    conformal = {
        "prediction_set": result.conformal_set,
        "set_size": len(result.conformal_set) if result.conformal_set else 0,
    }
    
    # Uncertainty report
    uncertainty = result.uncertainty_report
    
    # OOD detection
    ood = {
        "is_ood": result.ood_flag,
        "uncertainty_flag": result.uncertainty_flag,
    }
    
    # Nearest neighbors
    neighbors = result.nearest_neighbors
    
    return {
        "prediction": prediction,
        "conformal": conformal,
        "uncertainty": uncertainty,
        "ood": ood,
        "neighbors": neighbors,
        "timestamp": time.time(),
    }

result = run_inference("./query.jpg")
print(json.dumps(result["prediction"], indent=2))
```

---

## Step 4: Human Feedback Loop

```python
def handle_feedback(image_path: str, true_label: str, confidence: float = 1.0):
    """Process human feedback and update the model."""
    summary = learner.correct(
        image_path=image_path,
        true_label=true_label,
        confidence_weight=confidence,
    )
    
    # Periodic checkpoint
    if summary["total_corrections"] % 50 == 0:
        learner.save("production_checkpoint.json")
        print(f"Checkpoint saved ({summary['total_corrections']} corrections)")
    
    return summary

# Example: user corrects a misclassification
feedback = handle_feedback(
    image_path="./misclassified.jpg",
    true_label="actual_class",
    confidence=0.9,
)
print(f"Buffer size: {feedback['buffer_size']}")
print(f"Calibration updated: {feedback['calibration_updated']}")
print(f"Fine-tuned: {feedback['fine_tuned']}")
```

---

## Step 5: Memory Profiling — v0.2.0

AdaptShot v0.2.0 includes a `MemoryTracker` for production memory monitoring. Track memory usage at every pipeline stage:

```python
import tracemalloc
from adaptshot.profiling import MemoryTracker

# Initialize the memory tracker
tracker = MemoryTracker()
tracker.start()

# Track memory during support loading
with tracker.section("support_loading"):
    learner.load_support_images(image_paths, labels)

# Track memory during inference
with tracker.section("inference"):
    result = learner.predict("query.jpg")

# Track memory during correction
with tracker.section("correction"):
    learner.correct("query.jpg", true_label="cat", confidence_weight=0.9)

# Generate profiling report
report = tracker.get_report()
print(f"Peak memory (overall):      {report['peak_memory_mb']:.1f} MB")
print(f"Current memory:             {report['current_memory_mb']:.1f} MB")
print(f"Support loading peak:       {report['sections']['support_loading']['peak_mb']:.1f} MB")
print(f"Inference peak:             {report['sections']['inference']['peak_mb']:.1f} MB")
print(f"Correction peak:            {report['sections']['correction']['peak_mb']:.1f} MB")
```

### Lightweight Profiling with tracemalloc

```python
# Quick memory snapshot without MemoryTracker
tracemalloc.start()
result = learner.predict("query.jpg")
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Prediction memory — current: {current / 1024 / 1024:.1f} MB, peak: {peak / 1024 / 1024:.1f} MB")
```

### Memory Budget Monitoring

```python
# Set a memory budget and get alerts
budget_mb = 250  # AdaptShot's target: <250 MB

if report['peak_memory_mb'] > budget_mb:
    print(f"⚠️ Memory budget exceeded: {report['peak_memory_mb']:.0f} MB > {budget_mb} MB")
    print("   Consider:")
    print("   - Reducing max_buffer_size")
    print("   - Switching to mobilenet_v3_small backbone")
    print("   - Disabling explainability for non-review predictions")
```

---

## Step 6: Monitoring Dashboard

```python
def generate_monitoring_report():
    """Generate a production monitoring report."""
    calibration = learner.calibration_report()
    
    report = {
        "calibration": {
            "ece": calibration.get("debiased_ece", 0.0),
            "temperature": calibration.get("temperature", 1.0),
            "ood_threshold": calibration.get("ood_distance_threshold", float("inf")),
        },
        "buffer": {
            "support_size": calibration.get("support_size", 0),
            "prototype_count": calibration.get("prototype_count", 0),
        },
        "conformal": learner.conformal.get_calibration_summary(),
        "uncertainty": learner.uncertainty_q.get_ood_summary(),
        "act_thresholds": learner.act.get_all_thresholds(),
    }
    
    # v0.2.0: include historical penalty summary
    if hasattr(learner, 'explainability_engine'):
        report["historical_penalties"] = (
            learner.explainability_engine.get_penalty_summary()
        )
    
    return report

report = generate_monitoring_report()
print(json.dumps(report, indent=2, default=str))
```

---

## Step 7: Automated Quality Control

```python
def should_auto_accept(result, threshold=0.85):
    """Auto-accept high-confidence, low-uncertainty predictions."""
    return (
        result["prediction"]["confidence"] >= threshold
        and not result["ood"]["is_ood"]
        and not result["uncertainty"]["is_ood"]
        and result["prediction"]["act_action"] == "ACCEPT"
        and result["uncertainty"]["composite"] < 0.2
    )

def production_pipeline(image_path: str):
    """Full production pipeline with automated routing."""
    # Step 1: Inference
    result = run_inference(image_path)
    
    # Step 2: Quality check
    if should_auto_accept(result):
        return {
            "status": "AUTO_ACCEPTED",
            "prediction": result["prediction"]["class"],
            "confidence": result["prediction"]["confidence"],
        }
    
    # Step 3: Explain if uncertain
    explanation = learner.explain(image_path)
    
    return {
        "status": "NEEDS_REVIEW",
        "prediction": result["prediction"],
        "explanation": explanation.summary,
        "counterfactual": {
            "alternative": explanation.counterfactual.counterfactual_class,
            "margin": explanation.counterfactual.swap_required,
        },
    }

output = production_pipeline("./query.jpg")
print(f"Status: {output['status']}")
```

---

## Step 8: Periodic Maintenance

```python
def maintenance_cycle():
    """Run periodic maintenance tasks."""
    # Rebuild prototypes from current buffer
    learner._rebuild_prototypes()
    
    # Update OOD threshold
    learner._update_ood_threshold()
    
    # v0.2.0: clear backbone cache to reclaim memory
    learner.clear_backbone_cache()
    
    # Save with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    learner.save(f"checkpoints/production_{timestamp}.json")
    
    # Generate report
    report = generate_monitoring_report()
    
    # Alert on degradation
    if report["calibration"]["ece"] > 0.1:
        print("⚠️ ECE above 0.1 — calibration degrading")
    if report["buffer"]["support_size"] > config.max_buffer_size * 0.9:
        print("⚠️ Buffer at 90% capacity — consider pruning")
    
    # v0.2.0: check penalty trends
    if "historical_penalties" in report:
        trend = report["historical_penalties"].get("global_trend")
        if trend == "degrading":
            print("⚠️ Global penalty trend is DEGRADING — model may be drifting")
    
    return report

# Run daily or after significant data drift
maintenance_report = maintenance_cycle()
```

---

## Best Practices

1. **Save checkpoints regularly** — every 50-100 corrections
2. **Monitor ECE drift** — recalibrate if ECE exceeds 0.1
3. **Track OOD rate** — sudden increases indicate distribution shift
4. **Log all corrections** — for audit trails and model improvement
5. **Use `explain()` on rejected predictions** — helps diagnose systematic errors
6. **Profile memory weekly** — use MemoryTracker to catch leaks before they cause OOM
7. **Clear backbone cache periodically** — `clear_backbone_cache()` reclaims memory in long-running services
8. **Monitor penalty trends** — a "degrading" global trend signals potential model staleness

---

## v0.2.0 Hardening Summary

| Feature | v0.1.x | v0.2.0 |
|---------|--------|--------|
| Memory profiling | Manual tracemalloc only | MemoryTracker with section-level breakdowns |
| Backbone cache | Accumulated indefinitely | `clear_backbone_cache()` for production services |
| Penalty monitoring | None | Per-class trend analysis in monitoring dashboard |
| OOD threshold | Static | Adaptive, reported in monitoring |
| Checkpointing | Manual save/load | Same, with recommended periodic schedule |

---

## Next Steps

- [Architecture Deep-Dive](../guides/architecture-deep-dive.md)
- [Algorithm Theory](../guides/algorithm-theory.md)
- [Troubleshooting Guide](../guides/troubleshooting.md)
