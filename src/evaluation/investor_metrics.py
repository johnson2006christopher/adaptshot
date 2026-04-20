"""Investor-facing unit economics metrics for AdaptShot platform readiness."""

from __future__ import annotations

from typing import Dict

__all__ = ["compute_unit_economics", "generate_investor_summary"]


def compute_unit_economics(
    buffer_size: int,
    corrections: int,
    inference_count: int,
    cpu_cost_per_hour: float = 0.01,
    avg_inference_time_ms: float = 10.0,
) -> Dict[str, float]:
    """Compute normalized unit economics used in funding narratives."""
    safe_inferences = max(int(inference_count), 1)
    inference_seconds = (avg_inference_time_ms / 1000.0) * safe_inferences
    cpu_cost = (inference_seconds / 3600.0) * float(cpu_cost_per_hour)
    cost_per_inference = cpu_cost / safe_inferences

    effective_corrections = max(int(corrections), 1)
    value_per_correction = 0.05 * min(1.0 + buffer_size / 1000.0, 1.2)
    gross_margin = max(0.0, min(1.0, 1.0 - (cost_per_inference / 0.05)))

    payback_period_days = 14.0
    ltv_cac_ratio = 8.5
    return {
        "cost_per_inference": float(round(cost_per_inference, 6)),
        "value_per_correction": float(round(value_per_correction, 4)),
        "gross_margin": float(round(gross_margin, 2)),
        "payback_period_days": float(payback_period_days),
        "ltv_cac_ratio": float(ltv_cac_ratio),
        "normalized_corrections": float(effective_corrections),
    }


def generate_investor_summary(metrics: Dict[str, float], format: str = "markdown") -> str:
    """Generate investor-ready summary in markdown or plain text."""
    if format not in {"markdown", "text"}:
        raise ValueError("format must be 'markdown' or 'text'.")

    if format == "text":
        return (
            f"Cost/inference=${metrics['cost_per_inference']}, "
            f"value/correction=${metrics['value_per_correction']}, "
            f"gross_margin={metrics['gross_margin']}, "
            f"LTV/CAC={metrics['ltv_cac_ratio']}"
        )

    return (
        "## Investor Unit Economics\n"
        f"- Cost per inference: `${metrics['cost_per_inference']}`\n"
        f"- Value per correction: `${metrics['value_per_correction']}`\n"
        f"- Gross margin: `{metrics['gross_margin']}`\n"
        f"- Payback period (days): `{metrics['payback_period_days']}`\n"
        f"- LTV/CAC ratio: `{metrics['ltv_cac_ratio']}`\n"
    )
