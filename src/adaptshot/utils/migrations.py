"""Checkpoint migration helpers for AdaptShot persistence schemas."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "0.1.1"


def migrate_v0_1_0_to_v0_1_1(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v0.1.0 checkpoint payload to the v0.1.1 schema.

    Migration rules:
    - Preserve the original configuration and learned state.
    - Add the v0.1.1 schema version marker.
    - Normalize buffer fields so downstream validation can run safely.
    - Leave integrity metadata unset; v0.1.0 checkpoints did not ship a hash.

    Args:
        data: Raw checkpoint dictionary loaded from disk.

    Returns:
        A migrated checkpoint dictionary compatible with v0.1.1 loaders.
    """
    migrated = dict(data)
    migrated["schema_version"] = SCHEMA_VERSION

    buffer_state = dict(migrated.get("buffer", {}))
    buffer_state.setdefault("labels", [])
    buffer_state.setdefault("times", [])
    buffer_state.setdefault("uncertainties", [])
    buffer_state.setdefault("previews", [])
    migrated["buffer"] = buffer_state

    migrated.setdefault("calibration", {"temperature": 1.0, "ece_history": []})
    migrated.setdefault("act_thresholds", {})
    migrated.setdefault("is_initialized", False)
    return migrated
