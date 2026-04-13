# 🔬 Methodology: AdaptShot Few-Shot Pipeline

## 1. Problem Formulation
Modern vision models require thousands of labeled examples to generalize. In clinical, agricultural, and industrial domains, such datasets are often unavailable, expensive, or ethically constrained. AdaptShot addresses this by designing a **data-centric, uncertainty-aware pipeline** that:
- Learns effectively from ≤50 labeled examples per class
- Quantifies predictive uncertainty to enable safe deployment
- Provides a foundation for continuous improvement via human feedback

## 2. Architecture Overview
[Input Image] → [ResNet18 Backbone (Frozen)] → [512-dim Feature Vector]
                                               ↓
                                    [Linear Classifier Head (Trainable)]
                                               ↓
                                    [Class Probabilities + Entropy → Confidence]
- **Backbone**: ResNet18 pre-trained on ImageNet. Frozen to preserve universal visual features (edges, textures, shapes).
- **Head**: Single linear layer (`512 → 5`). Trained via Adam (lr=1e-3) for 5 epochs.
- **Uncertainty Engine**: Predictive entropy normalized to `[0, 1]` confidence score.

## 3. Mathematical Foundations
### 3.1 Transfer Learning Rationale
Given dataset $D = \{(x_i, y_i)\}_{i=1}^{N}$ with $N \ll 1000$, fine-tuning all parameters $\theta$ leads to severe overfitting:
$$\min_{\theta} \mathcal{L}(f_\theta(x), y) \rightarrow \text{high variance, poor generalization}$$
Instead, we optimize only the classifier head $\phi$ while keeping backbone weights $\theta_{backbone}$ fixed:
$$\min_{\phi} \mathcal{L}(g_\phi(h_{\theta_{backbone}}(x)), y)$$
This preserves pre-trained feature quality while adapting decision boundaries to the target domain.

### 3.2 Entropy-Based Confidence
For predicted probabilities $p = [p_1, \dots, p_C]$, entropy is:
$$H(p) = -\sum_{c=1}^{C} p_c \log(p_c)$$
Normalized confidence:
$$\text{Conf}(p) = 1 - \frac{H(p)}{\log(C)}$$
- $H(p) \approx 0 \Rightarrow \text{Conf} \approx 1$ (peaked distribution, high certainty)
- $H(p) \approx \log(C) \Rightarrow \text{Conf} \approx 0$ (uniform distribution, high uncertainty)

## 4. Design Choices & Rationale
| Choice | Rationale | Trade-off |
|--------|-----------|-----------|
| Freeze ResNet18 backbone | Prevents catastrophic overfitting on 50 images | Loses domain-specific feature adaptation |
| Entropy over MC Dropout (Day 1) | Zero additional compute, stable on CPU | Less robust to distribution shift |
| CPU fallback for P100 | Ensures reproducibility across hardware | Slower training (~5-8 min/5 epochs) |
| Strict seed control (42) | Guarantees identical results across runs | Limits exploration of variance |

## 5. Limitations & Known Constraints
- **Small validation set**: 25 test images → high variance in accuracy estimates
- **No temperature scaling**: Confidence scores are uncalibrated post-hoc
- **Static architecture**: Prototypical/Metric heads not yet implemented
- **Hardware dependency**: P100 CUDA 6.0 requires CPU fallback; newer GPUs recommended for production

## 6. Reproducibility Notes
- All experiments use `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`
- Dataset subsampling is deterministic via fixed seed
- Logs exported as JSON + CSV for independent verification
- Full environment captured in `requirements.txt` + `pyproject.toml`

---
*Last updated: Day 1/25 | Author: Johnson Hassan*
