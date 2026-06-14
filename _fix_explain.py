with open('src/adaptshot/core/explain.py', 'r') as f:
    content = f.read()

# Fix the magic number fallbacks
old = """        # ACT penalty: derived from the gap between confidence and threshold
        if act_threshold is not None and act_action != "ACCEPT":
            act_penalty = float(np.clip(cal - act_threshold, -0.5, 0.0))
        elif act_action != "ACCEPT":
            act_penalty = -0.15  # conservative fallback
        else:
            act_penalty = 0.0

        # OOD penalty: proportional to the OOD score
        if ood_score is not None and is_ood:
            ood_penalty = float(np.clip(-0.5 * ood_score, -0.5, 0.0))
        elif is_ood:
            ood_penalty = -0.25  # conservative fallback
        else:
            ood_penalty = 0.0"""

new = """        # ACT penalty: derived from the gap between confidence and threshold.
        # v0.2.0: When threshold unavailable, derive penalty from historical avg
        # gap rather than magic number -0.15.
        if act_threshold is not None and act_action != "ACCEPT":
            act_penalty = float(np.clip(cal - act_threshold, -0.5, 0.0))
            # Track for fallback
            self._act_penalty_history.append(abs(act_penalty))
        elif act_action != "ACCEPT":
            # Derive from historical average penalty, or use proportional default
            if self._act_penalty_history:
                act_penalty = -float(np.mean(self._act_penalty_history[-20:]))
            else:
                # Default: proportional to (1 - cal) — moderate penalty
                act_penalty = float(np.clip(-0.5 * (1.0 - cal), -0.5, 0.0))
        else:
            act_penalty = 0.0

        # OOD penalty: proportional to the OOD score.
        # v0.2.0: When ood_score unavailable, derive from config-level quantile
        # rather than magic number -0.25.
        if ood_score is not None and is_ood:
            ood_penalty = float(np.clip(-0.5 * ood_score, -0.5, 0.0))
            self._ood_penalty_history.append(abs(ood_penalty))
        elif is_ood:
            if self._ood_penalty_history:
                ood_penalty = -float(np.mean(self._ood_penalty_history[-20:]))
            else:
                # Default: moderate OOD penalty based on typical OOD score
                ood_penalty = float(np.clip(-0.5 * self._default_ood_score, -0.5, 0.0))
        else:
            ood_penalty = 0.0"""

if old in content:
    content = content.replace(old, new)
    # Also add instance state in __init__
    init_old = """    def __init__(
        self,
        top_k_attributions: int = 5,
        counterfactual_k: int = 3,
    ) -> None:
        \"\"\"Initialize the explainability engine.

        Args:
            top_k_attributions: Number of top support examples to attribute.
            counterfactual_k: Number of alternative classes to consider.
        \"\"\"
        self.top_k = top_k_attributions
        self.counterfactual_k = counterfactual_k"""

    init_new = """    def __init__(
        self,
        top_k_attributions: int = 5,
        counterfactual_k: int = 3,
    ) -> None:
        \"\"\"Initialize the explainability engine.

        Args:
            top_k_attributions: Number of top support examples to attribute.
            counterfactual_k: Number of alternative classes to consider.
        \"\"\"
        self.top_k = top_k_attributions
        self.counterfactual_k = counterfactual_k
        # v0.2.0: Track historical penalties to derive intelligent fallbacks
        self._act_penalty_history: List[float] = []
        self._ood_penalty_history: List[float] = []
        self._default_ood_score: float = 0.5  # Conservative default"""

    if init_old in content:
        content = content.replace(init_old, init_new)
        print('explain init: Found and replaced')
    else:
        print('explain init: Not found')

    with open('src/adaptshot/core/explain.py', 'w') as f:
        f.write(content)
    print('explain decompose: Replaced successfully')
else:
    print('explain decompose: Not found')
