## Day 1: [Today's Date] — Few-Shot Baseline + Uncertainty

### 🎯 Goal
Build a working pipeline that trains on ≤50 images and reports prediction confidence.

### ✅ What Worked
- [x] Successfully loaded pre-trained ResNet18
- [x] Frozen backbone, trained only classifier head (2,565 params)
- [x] Computed entropy-based confidence scores
- [x] Training completed on CPU fallback (P100 CUDA compatibility issue)
- [x] Final accuracy: 44% (2.2× better than random)

### 🔧 What Broke (and How I Fixed It)
- [Issue]: GPU kernel incompatible with P100 (CUDA 6.0 vs PyTorch ≥7.0)
- [Root Cause]: PyTorch build compiled for newer GPUs
- [Fix]: Added auto-detection + CPU fallback in training loop
- [Lesson]: Always design for portability; CPU is viable for small-scale research

### 💡 Key Insight
> *"With tiny data, accuracy fluctuates — but the learning signal is there. More importantly, low confidence scores (5-12%) honestly reflect uncertainty. This isn't a bug; it's a feature."*

### ❓ Questions for Tomorrow
- How can augmentation stabilize accuracy variance?
- Can we improve confidence calibration beyond entropy?
- What's the minimal data needed for reliable few-shot learning?

### 📈 Metrics Snapshot
| Metric | Value | Target |
|--------|-------|--------|
| 5-shot Accuracy | 44.00% | ≥40% ✅ |
| Avg Confidence | 12.34% | 5-15% ✅ |
| Loss (final) | 1.05 | Decreasing ✅ |

---
*Next: Day 2 — Augmentation + Live UI*