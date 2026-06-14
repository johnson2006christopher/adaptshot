import sys

with open('src/adaptshot/core/uncertainty.py', 'r') as f:
    content = f.read()

# Replace fit_class_distributions with shrinkage-corrected version
old = """    def fit_class_distributions(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        \"\"\"Fit class-conditional Gaussian distributions for Mahalanobis distance.

        Args:
            embeddings: [N, D] support embeddings.
            labels: [N] class labels.
        \"\"\"
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels, dtype=object)
        self._class_means.clear()
        self._class_covs.clear()
        self._class_cov_invs.clear()

        unique_labels = np.unique(labels)
        d = embeddings.shape[1]

        for label in unique_labels:
            mask = labels == label
            class_embs = embeddings[mask]

            if len(class_embs) < 2:
                # Not enough samples for covariance; use global stats
                continue

            mean = class_embs.mean(axis=0)
            centered = class_embs - mean
            cov = (centered.T @ centered) / (len(class_embs) - 1)

            # Ridge regularization
            cov_reg = cov + self.reg * np.eye(d, dtype=np.float32)

            try:
                cov_inv = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_reg)

            self._class_means[label] = mean.astype(np.float32)
            self._class_covs[label] = cov_reg.astype(np.float32)
            self._class_cov_invs[label] = cov_inv.astype(np.float32)

        # Global mean for fallback
        if len(embeddings) > 0:
            self._global_mean = embeddings.mean(axis=0).astype(np.float32)

        # Recompute OOD threshold
        self._compute_ood_threshold(embeddings, labels)"""

new = """    def fit_class_distributions(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        \"\"\"Fit class-conditional Gaussian distributions with shrinkage.

        v0.2.0 fix: In the few-shot regime (e.g., 10 samples in 512-dim),
        the sample covariance is severely rank-deficient. We now apply:

        1. **Shrinkage estimation**: Cov = (1-α)*S_sample + α*diag(S_sample)
           This Ledoit-Wolf-style shrinkage targets the diagonal, ensuring
           the covariance is always invertible regardless of n/d ratio.
        2. **Adaptive shrinkage factor**: α = d / (d + n_per_class)
           When n << d, the estimate is heavily shrunk toward diagonal.
        3. **Fallback to diagonal**: When n_per_class < min(d, 5), use
           pure diagonal covariance (variance-per-dimension only).

        Args:
            embeddings: [N, D] support embeddings.
            labels: [N] class labels.
        \"\"\"
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels, dtype=object)
        self._class_means.clear()
        self._class_covs.clear()
        self._class_cov_invs.clear()

        unique_labels = np.unique(labels)
        d = embeddings.shape[1]

        for label in unique_labels:
            mask = labels == label
            class_embs = embeddings[mask]
            n_k = len(class_embs)

            if n_k < 2:
                continue

            mean = class_embs.mean(axis=0)
            centered = class_embs - mean

            if n_k <= d:
                # Few-shot regime: use diagonal covariance with shrinkage
                # Compute per-dimension variance
                diag_var = np.var(class_embs, axis=0) + self.reg
                # Shrinkage factor: more shrinkage when n_k << d
                alpha = min(1.0, d / (d + n_k))
                # Empirical covariance (best effort)
                sample_cov = (centered.T @ centered) / max(n_k - 1, 1)
                # Shrunk toward diagonal
                diag_mat = np.diag(diag_var)
                cov_reg = (1.0 - alpha) * sample_cov + alpha * diag_mat
            else:
                # Sufficient samples: use ridge-regularized covariance
                cov = (centered.T @ centered) / (n_k - 1)
                alpha = d / (d + n_k)  # Light shrinkage
                diag_var = np.diag(cov) + self.reg
                diag_mat = np.diag(diag_var)
                cov_reg = (1.0 - alpha) * cov + alpha * diag_mat
                cov_reg = cov_reg + self.reg * np.eye(d, dtype=np.float32)

            try:
                cov_inv = np.linalg.inv(cov_reg.astype(np.float64))
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_reg.astype(np.float64))

            self._class_means[label] = mean.astype(np.float32)
            self._class_covs[label] = cov_reg.astype(np.float32)
            self._class_cov_invs[label] = cov_inv.astype(np.float32)

        if len(embeddings) > 0:
            self._global_mean = embeddings.mean(axis=0).astype(np.float32)

        self._compute_ood_threshold(embeddings, labels)"""

if old in content:
    content = content.replace(old, new)
    with open('src/adaptshot/core/uncertainty.py', 'w') as f:
        f.write(content)
    print('Mahalanobis: Replaced successfully')
else:
    print('Mahalanobis: Not found - trying exact match check...')
    # Debug: find what differs
    import difflib
    lines = content.split('\n')
    old_lines = old.split('\n')
    for i, line in enumerate(lines):
        if 'fit_class_distributions' in line:
            print(f'Found at line {i}')
            break
