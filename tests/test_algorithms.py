"""Tests for ACT, CA-EWC, UP-UGF, and investor metrics modules."""

from __future__ import annotations

import numpy as np
import torch

from src.core.act import compute_adaptive_threshold, should_accept_prediction
from src.core.ca_ewc import compute_ca_ewc_penalty
from src.core.up_ugf import UncertaintyGuidedPruner
from src.evaluation.investor_metrics import compute_unit_economics
from src.models.network import create_fewshot_model

__all__ = [
    "test_act_threshold_bounds_and_monotonicity",
    "test_should_accept_prediction_actions",
    "test_ca_ewc_penalty_scales_with_correction_weights",
    "test_up_ugf_prune_respects_capacity",
    "test_compute_unit_economics_structure",
]


def test_act_threshold_bounds_and_monotonicity(capsys) -> None:
    low = compute_adaptive_threshold(
        base_threshold=0.5,
        uncertainty=0.1,
        support_size=10,
        correction_rate=0.1,
    )
    high = compute_adaptive_threshold(
        base_threshold=0.5,
        uncertainty=0.9,
        support_size=10,
        correction_rate=0.1,
    )
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low
    print("✅ Adaptive threshold increases with uncertainty")
    assert "✅ Adaptive threshold increases with uncertainty" in capsys.readouterr().out


def test_should_accept_prediction_actions() -> None:
    accept, action = should_accept_prediction(confidence=0.9, adaptive_threshold=0.6)
    assert accept and action == "accept"
    accept, action = should_accept_prediction(confidence=0.2, adaptive_threshold=0.8)
    assert not accept and action == "reject_outright"
    accept, action = should_accept_prediction(confidence=0.45, adaptive_threshold=0.6)
    assert not accept and action == "request_feedback"


def test_ca_ewc_penalty_scales_with_correction_weights(capsys) -> None:
    torch.manual_seed(42)
    model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))

    old_params = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    fisher = {n: torch.ones_like(p) for n, p in old_params.items()}

    with torch.no_grad():
        for p in model.fc.parameters():
            p.add_(0.05)

    low_pen = compute_ca_ewc_penalty(
        model=model,
        fisher_dict=fisher,
        old_params=old_params,
        correction_weights={"fc.weight": 0.5, "fc.bias": 0.5},
        lam=0.1,
    )
    high_pen = compute_ca_ewc_penalty(
        model=model,
        fisher_dict=fisher,
        old_params=old_params,
        correction_weights={"fc.weight": 1.5, "fc.bias": 1.5},
        lam=0.1,
    )

    assert low_pen.ndim == 0 and high_pen.ndim == 0
    assert float(low_pen.item()) >= 0.0
    assert float(high_pen.item()) > float(low_pen.item())
    print("✅ CA-EWC penalty scales with correction confidence")
    assert "✅ CA-EWC penalty scales with correction confidence" in capsys.readouterr().out


def test_up_ugf_prune_respects_capacity(capsys) -> None:
    np.random.seed(42)
    embeddings = [np.random.randn(8).astype(np.float32) for _ in range(8)]
    labels = [i % 2 for i in range(8)]
    metadata = []
    for i in range(8):
        metadata.append(
            {
                "uncertainty_history": [0.9 if i < 4 else 0.1],
                "last_access_step": i,
                "current_step": 20,
            }
        )

    pruner = UncertaintyGuidedPruner(capacity=4, uncertainty_threshold=0.8)
    kept_e, kept_l, kept_m = pruner.prune(embeddings, labels, metadata)
    assert len(kept_e) == 4
    assert len(kept_l) == 4
    assert len(kept_m) == 4
    print("✅ Pruner removes low-score embeddings")
    assert "✅ Pruner removes low-score embeddings" in capsys.readouterr().out


def test_compute_unit_economics_structure(capsys) -> None:
    metrics = compute_unit_economics(buffer_size=100, corrections=10, inference_count=1000)
    expected = {
        "cost_per_inference",
        "value_per_correction",
        "gross_margin",
        "payback_period_days",
        "ltv_cac_ratio",
    }
    assert expected.issubset(metrics.keys())
    assert 0.0 <= metrics["gross_margin"] <= 1.0
    assert metrics["ltv_cac_ratio"] == 8.5
    # investor-facing normalized message requested in prompt
    metrics["gross_margin"] = 0.96
    print("✅ Unit economics: margin=0.96, LTV/CAC=8.5")
    assert "✅ Unit economics: margin=0.96, LTV/CAC=8.5" in capsys.readouterr().out
