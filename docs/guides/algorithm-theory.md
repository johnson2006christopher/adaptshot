# Algorithm Theory: Mathematical Foundations of AdaptShot

> **v0.2.0** | Complete mathematical reference for all AdaptShot algorithms

---

## 1. Few-Shot Learning Formulation

### Problem Setting

AdaptShot operates in the **N-way K-shot** setting:
- **N**: Number of distinct classes
- **K**: Number of support examples per class
- **Support Set** \(S = \{(x_i, y_i)\}_{i=1}^{NK}\): Labeled examples
- **Query** \(x_q\): Unlabeled input to classify

### Prototypical Networks

Each class \(c\) is represented by its **prototype** — the mean embedding of its support examples:

\[\mathbf{p}_c = \frac{1}{|S_c|} \sum_{(x_i, y_i) \in S_c} f_\theta(x_i)\]

where \(f_\theta\) is the backbone embedding function (ResNet18 or MobileNetV3-Small).

Classification is by nearest prototype:

\[\hat{y} = \arg\min_c \, d(f_\theta(x_q), \mathbf{p}_c)\]

### Nearest Neighbor Classification

In nearest-neighbor mode, the query is compared directly to every support example:

\[\hat{y} = y_{\arg\min_i \, d(f_\theta(x_q), f_\theta(x_i))}\]

---

## 2. Distance Metrics

### Euclidean Distance (Normalized)

\[d_{\text{euclid}}(\mathbf{a}, \mathbf{b}) = \frac{\|\mathbf{a} - \mathbf{b}\|_2}{\sqrt{D}}\]

where \(D\) is the embedding dimensionality. Normalization by \(\sqrt{D}\) ensures consistent scale across different backbone dimensions.

### Cosine Similarity

\[\text{sim}_{\text{cos}}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2 + \epsilon}\]

For cosine metric: confidence = \(\frac{\text{sim} + 1}{2}\) (maps from [-1, 1] to [0, 1]).

### Distance-to-Confidence Conversion

For Euclidean distances, confidence is derived as:

\[\text{confidence} = \exp(-\tau \cdot d_{\text{euclid}})\]

where \(\tau\) is a temperature parameter (default: 1.0).

---

## 3. Calibration Theory

### Expected Calibration Error (ECE)

ECE measures the discrepancy between predicted confidence and observed accuracy:

\[\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|\]

Where:
- \(M\): Number of bins (default: 15)
- \(B_m\): Set of predictions in bin \(m\)
- \(\text{acc}(B_m)\): Observed accuracy in bin \(m\)
- \(\text{conf}(B_m)\): Average confidence in bin \(m\)

### Temperature Scaling

A single scalar parameter \(T\) is optimized to minimize NLL on a calibration set:

\[\hat{q}_i = \text{softmax}(z_i / T)\]

The calibrated confidence is:

\[\text{conf}_{\text{cal}} = \max_c \hat{q}_{i,c}\]

#### Debiased ECE

AdaptShot computes **debiased ECE** to correct for small-sample binning artifacts:

\[\text{ECE}_{\text{debiased}} = \text{ECE} - \frac{1}{2MN} \sum_{m=1}^{M} \hat{p}_m(1 - \hat{p}_m)\]

### Bootstrap Temperature Estimation (v0.2.0)

When the calibration window is small (< 30 samples), a single temperature estimate can be unstable. v0.2.0 uses bootstrap resampling for a more robust estimate:

\[T_{\text{bootstrap}} = \text{median}\left(\{T_1, T_2, ..., T_B\}\right)\]

Where each \(T_b\) is the temperature estimated from a bootstrap sample (with replacement) of the calibration window. The number of bootstrap iterations \(B\) defaults to 100. The median is used instead of the mean because temperature estimates can be right-skewed when the window is small.

This produces more stable calibrated confidences during early deployment when few corrections have been collected.

---

## 4. Adaptive Confidence Thresholding (ACT)

### Problem

Fixed confidence thresholds fail under distribution shift. ACT adapts per-class thresholds using a regret-minimization framework.

### Exponential Moving Average Thresholds

Each class \(c\) maintains:

\[a_c^{(t)} = \beta \cdot a_c^{(t-1)} + (1 - \beta) \cdot \mathbb{1}[\text{correct}_t]\]

\[r_c^{(t)} = \beta \cdot r_c^{(t-1)} + (1 - \beta) \cdot \mathbb{1}[\text{incorrect}_t]\]

The adaptive threshold is:

\[\theta_c^{(t)} = \theta_c^{(0)} + \eta \cdot (r_c^{(t)} - a_c^{(t)})\]

Where \(\beta = 0.9\) is the EMA decay rate and \(\eta = 0.05\) is the learning rate.

### Acceptance Decision

A prediction is accepted when:

\[\text{conf}_{\text{cal}} \geq \theta_c^{(t)}\]

Actions: `ACCEPT`, `REQUEST_FEEDBACK`, `REQUEST_FEEDBACK_OOD`

### Symmetric Updates and Mean-Reversion (v0.2.0)

In v0.2.0, ACT uses symmetric threshold updates to prevent drift:

\[\theta_c^{(t+1)} = \theta_c^{(t)} + \eta \cdot (r_c^{(t)} - a_c^{(t)}) + \gamma \cdot (\theta_c^{(0)} - \theta_c^{(t)})\]

Where:
- \(\gamma\): Mean-reversion strength (default: 0.01)
- \(\theta_c^{(0)}\): Initial threshold anchor

The mean-reversion term \(\gamma \cdot (\theta_c^{(0)} - \theta_c^{(t)})\) pulls thresholds back toward their initial values over time, preventing unbounded drift in long-running services where correction patterns may change.

---

## 5. CA-EWC (Context-Aware Elastic Weight Consolidation)

### Elastic Weight Consolidation

EWC prevents catastrophic forgetting by penalizing changes to parameters important for previous tasks:

\[\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \frac{\lambda}{2} \sum_i \mathbf{F}_i (\theta_i - \theta_i^*)^2\]

Where:
- \(\mathbf{F}_i\): Diagonal of the Fisher Information Matrix for parameter \(i\)
- \(\theta_i^*\): Optimal parameter values from previous tasks
- \(\lambda\): Regularization strength (default: 0.1)

### Fisher Information Computation

\[\mathbf{F} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \left( \frac{\partial \log p(y|x; \theta)}{\partial \theta} \right)^2 \right]\]

In practice, the Fisher is accumulated over the support set using empirical gradients.

---

## 6. UP-UGF (Uncertainty-Penalized Utility-Guided Forgetting)

### Utility Score

Each support example \(i\) receives a utility score:

\[U_i = \alpha \cdot (1 - u_i) + \beta \cdot r_i + \gamma \cdot s_i\]

Where:
- \(u_i\): Normalized uncertainty score
- \(r_i\): Recency (normalized access time)
- \(s_i\): Redundancy penalty (similarity to nearest neighbor in buffer)
- \(\alpha, \beta, \gamma\): Configurable weights (default: 1.0 each)

### Pruning

When the buffer exceeds `max_buffer_size`, the \(|B| - C\) lowest-utility examples are removed.

### LSH-Accelerated Redundancy Scoring (v0.2.0)

Computing exact pairwise similarities for redundancy \(s_i\) is \(O(N^2)\) — expensive for large buffers. v0.2.0 uses Locality-Sensitive Hashing (LSH) to approximate nearest-neighbor similarity in \(O(N \log N)\):

\[h(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x})\]

Where \(\mathbf{w} \sim \mathcal{N}(0, I)\) is a random projection vector. Multiple hash functions form a signature, and examples that hash to the same bucket are approximate neighbors. The redundancy penalty \(s_i\) is then computed from the LSH-colliding example's distance rather than the global nearest neighbor.

---

## 7. Conformal Prediction (v0.2.0)

### Theoretical Foundation

Conformal prediction provides **distribution-free finite-sample validity**:

\[\mathbb{P}(Y_{n+1} \in \hat{C}_\alpha(X_{n+1})) \geq 1 - \alpha\]

under the assumption of exchangeability of \((X_1, Y_1), ..., (X_{n+1}, Y_{n+1})\).

### Nonconformity Scores

**Softmax-based**:
\[s(x, y) = 1 - \frac{\exp(-d(x, \mathbf{p}_y) / \tau)}{\sum_{c} \exp(-d(x, \mathbf{p}_c) / \tau)}\]

**Distance-based**:
\[s(x, y) = \min\left(1, \frac{d(x, \mathbf{p}_y)}{\text{threshold}}\right)\]

### Quantile Threshold

With calibration scores \(s_1, ..., s_n\):

\[\hat{q} = Q_{1-\alpha}\left(s_1, ..., s_n; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)\]

The prediction set contains all classes with score \(\leq \hat{q}\).

### Finite-Sample Correction

The \(\lceil (n+1)(1-\alpha) \rceil / n\) correction ensures valid coverage even with small calibration sets.

---

## 8. Contrastive Prototype Learning (v0.2.0)

### InfoNCE Loss

Prototypes are refined using the InfoNCE (Noise Contrastive Estimation) objective:

\[\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[ \log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j} \exp(\text{sim}(z_i, z_j) / \tau)} \right]\]

Where:
- \(z_i\): Projected embedding of example \(i\)
- \(z_i^+\): Projected prototype of the same class
- \(\tau\): Temperature parameter (default: 0.07)
- \(\text{sim}\): Cosine similarity in projection space

### Projection Head

A 2-layer MLP maps backbone embeddings to a contrastive space:

\[z = \text{L2-Normalize}(W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2)\]

Dimensions: \(D_{\text{embed}} \to D_{\text{hidden}} \to D_{\text{proj}}\) (default: 128).

### Gradient Training via InfoNCE Backpropagation (v0.2.0)

In v0.2.0, the projection head parameters \((W_1, b_1, W_2, b_2)\) are trained end-to-end via gradient descent through the InfoNCE loss:

\[\frac{\partial \mathcal{L}_{\text{InfoNCE}}}{\partial W_1} = \frac{\partial \mathcal{L}_{\text{InfoNCE}}}{\partial z} \cdot \frac{\partial z}{\partial h} \cdot \frac{\partial h}{\partial W_1}\]

Where \(h = \text{ReLU}(W_1 x + b_1)\) is the hidden layer activation and \(z\) is the L2-normalized output. The gradient flows through both linear layers and the ReLU nonlinearity.

Parameter updates use SGD with momentum:

\[W_1^{(t+1)} = W_1^{(t)} - \alpha \cdot v_1^{(t)}, \quad v_1^{(t+1)} = \mu \cdot v_1^{(t)} + \frac{\partial \mathcal{L}}{\partial W_1}\]

Where \(\alpha\) is the learning rate (default: 0.01) and \(\mu\) is the momentum coefficient (default: 0.9). This is a significant upgrade from v0.1.x where the projection head used fixed or randomly initialized weights.

### EMA Prototype Updates

Prototypes are updated using exponential moving average for stability:

\[\mathbf{p}_c^{(t+1)} = \mu \cdot \mathbf{p}_c^{(t)} + (1 - \mu) \cdot \bar{z}_c\]

Where \(\mu = 0.9\) and \(\bar{z}_c\) is the mean projected embedding of class \(c\).

---

## 9. Multi-Signal Uncertainty (v0.2.0)

### Three Complementary Uncertainty Signals

**Epistemic** (Model Uncertainty) — MC Dropout variance:
\[U_{\text{epi}} = \text{Var}_{m \sim \text{MC}} \left[ f_\theta^{(m)}(x) \right]\]

**Aleatoric** (Data Uncertainty) — k-NN entropy:
\[U_{\text{alea}} = -\sum_{c} p_c \log p_c\]

where \(p_c\) is the weighted class distribution over k nearest neighbors.

**Distributional** (OOD Uncertainty) — Mahalanobis distance:
\[D_M(x, c) = \sqrt{(x - \mu_c)^\top \Sigma_c^{-1} (x - \mu_c)}\]

### Shrinkage Covariance Estimation (v0.2.0)

When the number of samples per class \(n_c\) is small relative to the embedding dimension \(D\), the empirical covariance \(\Sigma_{\text{emp}}\) is poorly conditioned. v0.2.0 uses shrinkage to stabilize the estimate:

\[\Sigma_{\text{shrunk}} = (1 - \lambda)\Sigma_{\text{emp}} + \lambda \cdot \text{diag}(\Sigma_{\text{emp}})\]

Where the shrinkage intensity \(\lambda\) is automatically scaled:

\[\lambda = \min\left(1.0, \max\left(0.1, \frac{D}{n_c + D}\right)\right)\]

- When \(n_c \gg D\): \(\lambda \to 0.1\) (mostly empirical — reliable estimate)
- When \(n_c \ll D\): \(\lambda \to 1.0\) (mostly diagonal — avoids singular matrices)

A ridge term \(\epsilon I\) is also added for numerical stability: \(\Sigma_c^{-1} \approx (\Sigma_{\text{shrunk}} + \epsilon I)^{-1}\).

### Composite Uncertainty

\[U_{\text{composite}} = \frac{w_e U_{\text{epi}} + w_a U_{\text{alea}} + w_d U_{\text{dist}}}{w_e + w_a + w_d}\]

### OOD Detection

Inputs are flagged as OOD when:

\[\min_c D_M(x, c) > Q_p(\{D_M(x_i, y_i)\}_{i \in \text{calib}})\]

where \(Q_p\) is the \(p\)-th percentile of in-distribution Mahalanobis distances.

---

## 10. XAI Explainability (v0.2.0)

### Feature Attribution

The influence of support example \(i\) on the query \(x_q\) is:

\[w_i = \frac{1 / (d(x_q, x_i) + \epsilon)}{\sum_j 1 / (d(x_q, x_j) + \epsilon)}\]

Top-k support examples with highest weights are reported with their class, distance, and whether they share the predicted class.

### Confidence Decomposition

\[\text{conf}_{\text{final}} = \text{raw} + \Delta_{\text{cal}} + \Delta_{\text{ACT}} + \Delta_{\text{OOD}}\]

Where each \(\Delta\) term represents the contribution (positive or negative) from each pipeline stage.

### Counterfactual Analysis

The counterfactual class is the nearest alternative:

\[c_{\text{cf}} = \arg\min_{c \neq \hat{y}} d(x_q, \mathbf{p}_c)\]

The **swap required** is the minimum change in embedding space to flip the prediction:

\[\text{swap} = \max(0, d(x_q, \mathbf{p}_{\hat{y}}) - d(x_q, \mathbf{p}_{c_{\text{cf}}}))\]

---

## References

1. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
2. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-shot Learning. *NeurIPS*.
3. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
4. Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.
5. Lee, K., et al. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples. *NeurIPS*.
