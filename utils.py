"""Utilities for the optional AdaptShot Studio GUI.

The helpers in this module keep the UI thin: they validate uploads, build
learner configuration, persist lightweight session snapshots, and extract
truthful diagnostics from the existing AdaptShot runtime objects.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import platform
import resource
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from ..config.settings import AdaptShotConfig
from ..core.learner import FewShotLearner
from ..utils.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)

SESSION_PATH = Path.home() / ".adaptshot" / "studio_session.json"
EXPORT_ROOT = Path.home() / ".adaptshot" / "studio_exports"
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp", ".tif", ".tiff"}


def default_config_values() -> dict[str, Any]:
    """Return the default studio configuration values.

    Returns:
        Dictionary of UI-friendly configuration values.
    """

    return {
        "backbone": "resnet18",
        "eco_mode": False,
        "max_buffer_size": 100,
        "calibration_method": "temperature",
        "seed": 42,
    }


def build_config(values: Mapping[str, Any]) -> AdaptShotConfig:
    """Build an AdaptShotConfig from studio form values.

    Args:
        values: Mapping of form values keyed by config field name.

    Returns:
        Validated AdaptShotConfig instance.
    """

    backbone = cast(Literal["resnet18", "mobilenet_v3_small"], str(values.get("backbone", "resnet18")))
    calibration_method = cast(
        Literal["temperature", "scaling_binning", "conformal", "none"],
        str(values.get("calibration_method", "temperature")),
    )

    return AdaptShotConfig(
        backbone=backbone,
        device="cpu",
        seed=int(values.get("seed", 42)),
        eco_mode=bool(values.get("eco_mode", False)),
        max_buffer_size=int(values.get("max_buffer_size", 100)),
        calibration_method=calibration_method,
    )


def config_to_json(config: AdaptShotConfig) -> str:
    """Serialize a config as pretty JSON."""

    return json.dumps(asdict(config), indent=2, sort_keys=True)


def estimate_runtime_profile(config: AdaptShotConfig, support_count: int = 0) -> dict[str, float]:
    """Estimate idle RAM and inference latency for display purposes.

    These are heuristic estimates, not benchmark claims.
    """

    backbone_base = 14.0 if config.backbone == "mobilenet_v3_small" else 20.0
    eco_bonus = -4.0 if config.eco_mode else 0.0
    buffer_ram = max(0.5, float(config.max_buffer_size) * 0.03)
    idle_ram_mb = 92.0 + backbone_base + eco_bonus + buffer_ram
    support_ram_mb = idle_ram_mb + float(max(0, support_count)) * 0.04
    latency_ms = 62.0 + (11.0 if config.backbone == "resnet18" else 6.0)
    latency_ms += max(0.0, float(support_count) * 0.35)
    if config.eco_mode:
        latency_ms *= 0.86

    return {
        "idle_ram_mb": round(idle_ram_mb, 2),
        "support_ram_mb": round(support_ram_mb, 2),
        "estimated_inference_ms": round(latency_ms, 2),
    }


def runtime_health_snapshot() -> dict[str, Any]:
    """Collect local-only runtime diagnostics."""

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "offline_only": True,
        "rss_mb": round(_process_rss_mb(), 2),
    }


def _process_rss_mb() -> float:
    """Return current process RSS in MB."""

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def validate_file_bundle(paths: Sequence[str | Path], max_total_mb: int = 100) -> list[Path]:
    """Validate uploaded file size and existence constraints."""

    normalized = [Path(path) for path in paths]
    if not normalized:
        raise ConfigValidationError("Upload at least one image before continuing.")

    total_bytes = 0
    for path in normalized:
        if not path.exists():
            raise ConfigValidationError(f"Uploaded file not found: {path}")
        total_bytes += path.stat().st_size

    total_mb = total_bytes / (1024.0 * 1024.0)
    if total_mb > float(max_total_mb):
        raise ConfigValidationError(
            f"Uploaded files total {total_mb:.1f} MB, which exceeds the {max_total_mb} MB limit."
        )

    return normalized


def discover_images_in_folder(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Discover image files inside a local folder.

    Args:
        folder: Folder path to scan.
        recursive: Whether to traverse subfolders.

    Returns:
        Sorted list of image paths.
    """

    folder_path = Path(folder).expanduser()
    if not folder_path.exists():
        raise ConfigValidationError(f"Folder not found: {folder_path}")
    if not folder_path.is_dir():
        raise ConfigValidationError(f"Expected a folder path, got: {folder_path}")

    iterator = folder_path.rglob("*") if recursive else folder_path.glob("*")
    images = [path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def collect_image_sources(
    file_paths: Sequence[str | Path],
    folder_text: str = "",
    recursive: bool = True,
) -> list[Path]:
    """Combine uploaded files with folder-based imports.

    Args:
        file_paths: Uploaded files.
        folder_text: One or more local folder paths separated by commas or new lines.
        recursive: Whether to search folders recursively.

    Returns:
        Normalized list of image paths.
    """

    collected = [Path(path).expanduser() for path in file_paths]
    folder_candidates = [part.strip() for part in folder_text.replace("\n", ",").split(",") if part.strip()]
    for folder in folder_candidates:
        collected.extend(discover_images_in_folder(folder, recursive=recursive))

    unique: dict[str, Path] = {}
    for path in collected:
        unique[str(path.resolve())] = path.resolve()
    return sorted(unique.values(), key=lambda path: str(path))


def infer_labels(paths: Sequence[Path], strategy: str, manual_labels: str = "") -> list[str]:
    """Infer labels from uploaded images."""

    if strategy == "manual":
        labels = [label.strip() for label in _split_labels(manual_labels)]
        if len(labels) == 1 and len(paths) > 1:
            labels = labels * len(paths)
        if len(labels) != len(paths):
            raise ConfigValidationError(
                "Manual labels must provide exactly one label per image, or a single label to apply to all."
            )
        return labels

    if strategy == "folder":
        inferred = [path.parent.name.strip() or path.stem for path in paths]
        return inferred

    if strategy == "stem":
        return [path.stem for path in paths]

    raise ConfigValidationError(f"Unknown label strategy '{strategy}'.")


def summarize_folder_imports(paths: Sequence[Path]) -> dict[str, int]:
    """Summarize how many images were imported from each folder."""

    summary: dict[str, int] = {}
    for path in paths:
        key = str(path.parent)
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items(), key=lambda item: item[0]))


class DeploymentSimilarityModel(torch.nn.Module):
    """TorchScript/ONNX-friendly similarity matcher for deployment exports."""

    support_embeddings: Any
    support_labels: Any
    temperature: Any

    def __init__(self, support_embeddings: np.ndarray, support_labels: Sequence[int], temperature: float) -> None:
        """Store support embeddings and class indices as buffers."""

        super().__init__()
        support_tensor = torch.as_tensor(support_embeddings, dtype=torch.float32)
        label_tensor = torch.as_tensor(list(support_labels), dtype=torch.long)
        self.register_buffer("support_embeddings", support_tensor)
        self.register_buffer("support_labels", label_tensor)
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, query_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return top label index, raw similarity, and calibrated confidence."""

        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        query = F.normalize(query_embedding.float(), dim=-1)
        support = F.normalize(self.support_embeddings.float(), dim=-1)
        similarities = torch.matmul(query, support.transpose(0, 1))
        top_scores, top_indices = torch.max(similarities, dim=-1)
        predicted_labels = self.support_labels[top_indices]
        calibrated = torch.clamp((top_scores + 1.0) * 0.5, 0.0, 1.0)
        return predicted_labels, top_scores, calibrated


def build_deployment_model(learner: FewShotLearner) -> DeploymentSimilarityModel:
    """Create a deployment-friendly similarity matcher from the live learner."""

    embeddings = np.asarray(getattr(learner, "_sim_embeddings", []), dtype=np.float32)
    if embeddings.size == 0:
        raise ConfigValidationError("Cannot export deployment model without support embeddings.")

    label_to_idx = dict(getattr(learner, "_label_to_idx", {}))
    support_labels_raw = list(getattr(learner, "_sim_labels", []))
    support_labels = [int(label_to_idx[label]) for label in support_labels_raw]
    temperature = float(learner.calibrator.current_temperature)
    return DeploymentSimilarityModel(embeddings, support_labels, temperature)


def export_deployment_bundle(learner: FewShotLearner, export_format: str, session: StudioSession | None = None) -> Path:
    """Export a deployment bundle in native, TorchScript, or ONNX format."""

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = EXPORT_ROOT / f"adaptshot_studio_{export_format}_{stamp}"
    checkpoint_path = stem.with_suffix(".json")
    report_path = stem.with_suffix(".md")
    usage_path = stem.with_name(f"{stem.name}_usage.txt")
    labels_path = stem.with_name(f"{stem.name}_labels.json")
    zip_path = stem.with_suffix(".zip")

    learner.save(str(checkpoint_path))
    report_path.write_text(build_report_markdown(learner, session=session), encoding="utf-8")
    usage_path.write_text(
        "Install command:\n"
        "  pip install adaptshot\n\n"
        "Native checkpoint load example:\n"
        "  from adaptshot.core.learner import FewShotLearner\n"
        "  learner = FewShotLearner.load('PATH_TO_CHECKPOINT.json')\n",
        encoding="utf-8",
    )

    label_to_idx = dict(getattr(learner, "_label_to_idx", {}))
    idx_to_label = {int(idx): str(label) for label, idx in label_to_idx.items()}
    labels_path.write_text(json.dumps(idx_to_label, indent=2, sort_keys=True), encoding="utf-8")

    extra_files: list[Path] = [
        checkpoint_path,
        checkpoint_path.with_suffix(".embeddings.npy"),
        checkpoint_path.with_suffix(".head.pt"),
        report_path,
        usage_path,
        labels_path,
    ]

    if export_format == "native":
        pass
    elif export_format == "torchscript":
        model = build_deployment_model(learner)
        model.eval()
        support_embeddings = model.support_embeddings
        example_input = torch.zeros(1, int(support_embeddings.shape[1]), dtype=torch.float32)
        scripted = torch.jit.trace(model, (example_input,))  # type: ignore[no-untyped-call]
        torchscript_path = stem.with_suffix(".pt")
        scripted.save(str(torchscript_path))
        extra_files.append(torchscript_path)
    elif export_format == "onnx":
        model = build_deployment_model(learner)
        model.eval()
        onnx_path = stem.with_suffix(".onnx")
        support_embeddings = model.support_embeddings
        dummy = torch.zeros(1, int(support_embeddings.shape[1]), dtype=torch.float32)
        try:
            torch.onnx.export(
                model,
            (dummy,),
                str(onnx_path),
                input_names=["query_embedding"],
                output_names=["predicted_label", "raw_similarity", "calibrated_confidence"],
                dynamic_axes={"query_embedding": {0: "batch"}, "predicted_label": {0: "batch"}, "raw_similarity": {0: "batch"}, "calibrated_confidence": {0: "batch"}},
                opset_version=18,
            )
        except Exception as exc:
            raise ConfigValidationError(
                f"ONNX export failed. Ensure the runtime supports torch.onnx.export. Details: {exc}"
            ) from exc
        extra_files.append(onnx_path)
    else:
        raise ConfigValidationError(f"Unknown export format '{export_format}'.")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in extra_files:
            if file_path.exists():
                archive.write(file_path, arcname=file_path.name)

    return zip_path


def load_project_bundle(bundle_path: str | Path) -> FewShotLearner:
    """Load a native checkpoint bundle or extracted checkpoint JSON."""

    target = Path(bundle_path)
    if not target.exists():
        raise ConfigValidationError(f"Project bundle not found: {target}")

    if target.is_dir():
        candidate_json = next(iter(sorted(target.glob("*.json"))), None)
        if candidate_json is None:
            raise ConfigValidationError(f"No checkpoint JSON found in directory: {target}")
        return FewShotLearner.load(str(candidate_json))

    if target.suffix.lower() == ".zip":
        with zipfile.ZipFile(target, mode="r") as archive:
            extract_dir = Path.home() / ".adaptshot" / "studio_imports" / f"{target.stem}_{int(time.time())}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            archive.extractall(extract_dir)
            candidate_json = next(iter(sorted(extract_dir.glob("*.json"))), None)
            if candidate_json is None:
                raise ConfigValidationError("The bundle zip did not contain a checkpoint JSON file.")
            return FewShotLearner.load(str(candidate_json))

    if target.suffix.lower() == ".json":
        return FewShotLearner.load(str(target))

    raise ConfigValidationError(
        "Unsupported project bundle format. Use a .zip export bundle or a checkpoint .json file."
    )


def benchmark_smoke_metrics(config: AdaptShotConfig) -> dict[str, Any]:
    """Run the existing deterministic energy smoke test for truthful metrics."""

    try:
        benchmark_module = importlib.import_module("benchmarks.energy_profile")
    except Exception as exc:  # pragma: no cover - benchmark module should exist in this repo
        raise ConfigValidationError(f"Benchmark smoke test could not be imported: {exc}") from exc

    run_smoke_test = cast(Any, benchmark_module.run_smoke_test)
    return cast(dict[str, Any], run_smoke_test(config))


def _split_labels(text: str) -> list[str]:
    """Split a user-provided label specification into individual labels."""

    if not text.strip():
        return []
    normalized = text.replace("\n", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def file_metadata_rows(paths: Sequence[Path], labels: Sequence[str | int]) -> list[dict[str, Any]]:
    """Build a metadata table for support uploads."""

    rows: list[dict[str, Any]] = []
    for path, label in zip(paths, labels):
        rows.append(
            {
                "file": path.name,
                "path": str(path),
                "label": str(label),
                "size_mb": round(path.stat().st_size / (1024.0 * 1024.0), 3),
                "validation": _validate_image_preview(path),
            }
        )
    return rows


def gallery_items(paths: Sequence[Path], labels: Sequence[str | int]) -> list[tuple[str, str]]:
    """Build gallery items for Gradio image previews."""

    return [(str(path), f"{path.name} | {label}") for path, label in zip(paths, labels)]


def _validate_image_preview(path: Path) -> str:
    """Check whether a file can be opened as an RGB image."""

    try:
        with Image.open(path) as image:
            image.verify()
        return "ready"
    except (UnidentifiedImageError, OSError):
        return "invalid image"


def confidence_bucket(value: float) -> str:
    """Convert a confidence score into a short user-facing bucket label."""

    if value >= 0.85:
        return "high"
    if value >= 0.60:
        return "medium"
    return "low"


def prediction_rows(
    paths: Sequence[Path],
    results: Sequence[Any],
    latencies_ms: Sequence[float],
) -> list[dict[str, Any]]:
    """Format inference outputs for tables and downloads."""

    rows: list[dict[str, Any]] = []
    for index, (path, result, latency_ms) in enumerate(zip(paths, results, latencies_ms), start=1):
        calibrated_confidence = float(getattr(result, "calibrated_confidence", 0.0))
        rows.append(
            {
                "id": f"pred-{index}",
                "file": path.name,
                "path": str(path),
                "prediction": str(getattr(result, "prediction", "")),
                "raw_confidence": round(float(getattr(result, "raw_confidence", 0.0)), 4),
                "calibrated_confidence": round(calibrated_confidence, 4),
                "confidence_bucket": confidence_bucket(calibrated_confidence),
                "uncertainty_flag": bool(getattr(result, "uncertainty_flag", False)),
                "act_action": str(getattr(result, "act_action", "")),
                "neighbor_idx": int(getattr(result, "neighbor_idx", -1)),
                "latency_ms": round(float(latency_ms), 2),
            }
        )
    return rows


def prediction_choices(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Build human-readable dropdown choices for prediction selection."""

    return [f"{row['id']} | {row['file']} | {row['prediction']}" for row in rows]


def choose_prediction_row(rows: Sequence[Mapping[str, Any]], selection: str) -> Mapping[str, Any] | None:
    """Select a prediction row by dropdown value or fallback to the latest row."""

    if not rows:
        return None
    for row in rows:
        if selection and str(row.get("id")) in selection:
            return row
    return rows[-1]


def summarize_labels(labels: Sequence[str | int]) -> dict[str, int]:
    """Return a class distribution summary."""

    counts: dict[str, int] = {}
    for label in labels:
        label_text = str(label)
        counts[label_text] = counts.get(label_text, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def build_support_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """Build a short summary string for support/class distribution."""

    if not rows:
        return "No support images loaded yet."
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    parts = [f"{label}: {count}" for label, count in counts.items()]
    return f"Classes: {len(counts)} | " + ", ".join(parts)


def build_prediction_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """Build a short summary string for inference output."""

    if not rows:
        return "No predictions yet."
    uncertain = sum(1 for row in rows if bool(row.get("uncertainty_flag")))
    return f"Predictions: {len(rows)} | Uncertain: {uncertain}"


def buffer_snapshot(learner: FewShotLearner) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect the learner's current replay buffer state."""

    embeddings = np.asarray(getattr(learner, "_sim_embeddings", []), dtype=np.float32)
    labels = list(getattr(learner, "_sim_labels", []))
    uncertainties = np.asarray(getattr(learner, "_sim_uncertainties", []), dtype=np.float32)
    access_times = np.asarray(getattr(learner, "_sim_access_times", []), dtype=np.float64)

    if embeddings.size == 0:
        return [], {
            "buffer_size": 0,
            "memory_mb": 0.0,
            "retained_count": 0,
            "pruned_count": 0,
        }

    scores = learner.pruner.compute_scores(embeddings, uncertainties, access_times)
    norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim_matrix = norms @ norms.T
    np.fill_diagonal(sim_matrix, -1.0)
    max_sim = np.max(sim_matrix, axis=1)
    now = time.time()

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        embedding = np.ascontiguousarray(embeddings[index], dtype=np.float32)
        rows.append(
            {
                "embedding_hash": hashlib.sha1(embedding.tobytes()).hexdigest()[:12],
                "label": str(label),
                "uncertainty": round(float(uncertainties[index]), 4) if len(uncertainties) else 0.0,
                "recency_seconds": round(float(max(0.0, now - float(access_times[index]))), 2)
                if len(access_times)
                else 0.0,
                "redundancy_score": round(float(max(0.0, 1.0 - max_sim[index])), 4),
                "retention_score": round(float(scores[index]), 4),
            }
        )

    summary = {
        "buffer_size": len(rows),
        "memory_mb": round(float(embeddings.nbytes) / (1024.0 * 1024.0), 3),
        "retained_count": len(rows),
        "pruned_count": max(0, len(rows) - learner.config.max_buffer_size),
    }
    return rows, summary


def estimate_buffer_memory_mb(learner: FewShotLearner) -> float:
    """Estimate the memory footprint of the in-memory support buffer."""

    embeddings = np.asarray(getattr(learner, "_sim_embeddings", []), dtype=np.float32)
    if embeddings.size == 0:
        return 0.0
    return round(float(embeddings.nbytes) / (1024.0 * 1024.0), 3)


def build_report_markdown(learner: FewShotLearner, session: StudioSession | None = None) -> str:
    """Build a truthful markdown summary for export."""

    support_size = len(getattr(learner, "_sim_embeddings", []))
    thresholds = learner.act.get_all_thresholds()
    ece = learner.calibrator.current_ece
    health = runtime_health_snapshot()
    benchmark_lines = ["- Benchmark smoke test: not run yet."]
    if session is not None and session.benchmark_snapshot:
        benchmark_lines = [
            f"- Benchmark accuracy: {float(session.benchmark_snapshot.get('accuracy', 0.0)):.4f}",
            f"- Benchmark avg latency: {float(session.benchmark_snapshot.get('latency_avg_s', 0.0)) * 1000.0:.2f} ms",
            f"- Benchmark p95 latency: {float(session.benchmark_snapshot.get('latency_p95_s', 0.0)) * 1000.0:.2f} ms",
            f"- Benchmark energy estimate: {float(session.benchmark_snapshot.get('joules_estimate', 0.0)):.4f} J",
            f"- Benchmark CO2 estimate: {float(session.benchmark_snapshot.get('co2_g_estimate', 0.0)):.6f} g",
        ]
    lines = [
        "# AdaptShot Studio Export Report",
        "",
        "## Configuration",
        f"- Backbone: {learner.config.backbone}",
        f"- Device: {learner.config.device}",
        f"- Eco mode: {learner.config.eco_mode}",
        f"- Max buffer size: {learner.config.max_buffer_size}",
        f"- Calibration method: {learner.config.calibration_method}",
        f"- Seed: {learner.config.seed}",
        "",
        "## Current State",
        f"- Support examples: {support_size}",
        f"- Current temperature: {learner.calibrator.current_temperature:.4f}",
        f"- Current ECE: {ece:.4f}",
        f"- ACT thresholds tracked: {len(thresholds)}",
        "",
        "## Metrics",
        f"- Idle RSS: {health['rss_mb']:.2f} MB",
        *benchmark_lines,
    ]

    if session is not None:
        lines.extend(
            [
                "",
                "## Session",
                f"- Loaded support files: {len(session.support_paths)}",
                f"- Recorded predictions: {len(session.prediction_rows)}",
                f"- Log lines: {len(session.logs)}",
            ]
        )

    return "\n".join(lines)


def export_native_bundle(learner: FewShotLearner, session: StudioSession | None = None) -> Path:
    """Export the native AdaptShot checkpoint bundle as a zip file."""

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = EXPORT_ROOT / f"adaptshot_studio_{stamp}"
    checkpoint_path = stem.with_suffix(".json")
    report_path = stem.with_suffix(".md")
    usage_path = stem.with_name(f"{stem.name}_usage.txt")
    zip_path = stem.with_suffix(".zip")

    learner.save(str(checkpoint_path))
    report_path.write_text(build_report_markdown(learner, session=session), encoding="utf-8")
    usage_path.write_text(
        "Install command:\n"
        "  pip install adaptshot\n\n"
        "Load checkpoint example:\n"
        "  from adaptshot.core.learner import FewShotLearner\n"
        "  learner = FewShotLearner.load('PATH_TO_CHECKPOINT.json')\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in (
            checkpoint_path,
            checkpoint_path.with_suffix(".embeddings.npy"),
            checkpoint_path.with_suffix(".head.pt"),
            report_path,
            usage_path,
        ):
            if file_path.exists():
                archive.write(file_path, arcname=file_path.name)

    return zip_path


def save_log_file(log_lines: Sequence[str], path: str | Path | None = None) -> Path:
    """Persist the studio log buffer to disk."""

    target = Path(path) if path is not None else EXPORT_ROOT / "adaptshot_studio.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
    return target


def append_log(session: StudioSession, message: str, level: str = "info") -> None:
    """Append a timestamped log entry to the studio session."""

    timestamp = time.strftime("%H:%M:%S")
    session.logs.append(f"[{timestamp}] {level.upper()}: {message}")


def log_text(session: StudioSession) -> str:
    """Render the current session log as plain text."""

    return "\n".join(session.logs[-200:])


def persist_session_snapshot(session: StudioSession, path: str | Path = SESSION_PATH) -> Path:
    """Persist a lightweight session snapshot to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(session.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_session_snapshot(path: str | Path = SESSION_PATH) -> StudioSession:
    """Load a lightweight session snapshot from disk if it exists."""

    target = Path(path)
    if not target.exists():
        return StudioSession()
    data = json.loads(target.read_text(encoding="utf-8"))
    return StudioSession.from_snapshot(data)


@dataclass
class StudioSession:
    """Mutable per-browser-session state for AdaptShot Studio."""

    config_values: dict[str, Any] = field(default_factory=default_config_values)
    learner: FewShotLearner | None = None
    support_paths: list[str] = field(default_factory=list)
    support_labels: list[str | int] = field(default_factory=list)
    support_rows: list[dict[str, Any]] = field(default_factory=list)
    query_paths: list[str] = field(default_factory=list)
    prediction_rows: list[dict[str, Any]] = field(default_factory=list)
    prediction_choices: list[str] = field(default_factory=list)
    selected_prediction_id: str = ""
    ece_history: list[float] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    last_error: str = ""
    export_paths: list[str] = field(default_factory=list)
    buffer_rows: list[dict[str, Any]] = field(default_factory=list)
    buffer_summary: dict[str, Any] = field(default_factory=dict)
    health_snapshot: dict[str, Any] = field(default_factory=runtime_health_snapshot)
    benchmark_snapshot: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable session snapshot."""

        return {
            "config_values": dict(self.config_values),
            "support_paths": list(self.support_paths),
            "support_labels": [str(label) for label in self.support_labels],
            "support_rows": list(self.support_rows),
            "query_paths": list(self.query_paths),
            "prediction_rows": list(self.prediction_rows),
            "prediction_choices": list(self.prediction_choices),
            "selected_prediction_id": self.selected_prediction_id,
            "ece_history": list(self.ece_history),
            "logs": list(self.logs),
            "last_error": self.last_error,
            "export_paths": list(self.export_paths),
            "buffer_rows": list(self.buffer_rows),
            "buffer_summary": dict(self.buffer_summary),
            "health_snapshot": dict(self.health_snapshot),
            "benchmark_snapshot": dict(self.benchmark_snapshot),
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> StudioSession:
        """Restore a session snapshot from disk."""

        session = cls()
        session.config_values = dict(payload.get("config_values", default_config_values()))
        session.support_paths = [str(path) for path in payload.get("support_paths", [])]
        session.support_labels = [str(label) for label in payload.get("support_labels", [])]
        session.support_rows = [dict(row) for row in payload.get("support_rows", [])]
        session.query_paths = [str(path) for path in payload.get("query_paths", [])]
        session.prediction_rows = [dict(row) for row in payload.get("prediction_rows", [])]
        session.prediction_choices = [str(choice) for choice in payload.get("prediction_choices", [])]
        session.selected_prediction_id = str(payload.get("selected_prediction_id", ""))
        session.ece_history = [float(value) for value in payload.get("ece_history", [])]
        session.logs = [str(line) for line in payload.get("logs", [])]
        session.last_error = str(payload.get("last_error", ""))
        session.export_paths = [str(path) for path in payload.get("export_paths", [])]
        session.buffer_rows = [dict(row) for row in payload.get("buffer_rows", [])]
        session.buffer_summary = dict(payload.get("buffer_summary", {}))
        session.health_snapshot = dict(payload.get("health_snapshot", runtime_health_snapshot()))
        session.benchmark_snapshot = dict(payload.get("benchmark_snapshot", {}))
        return session


def create_learner(config_values: Mapping[str, Any]) -> FewShotLearner:
    """Create a CPU-only FewShotLearner from studio config values."""

    config = build_config(config_values)
    return FewShotLearner(config=config)
