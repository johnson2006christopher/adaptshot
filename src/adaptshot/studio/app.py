"""AdaptShot Studio: an optional offline Gradio workspace.

The studio is a lightweight browser UI that wraps the existing AdaptShot
learner, calibration, ACT, feedback, and buffer management surfaces. It avoids
inventing backend behavior; unsupported exports and metrics are labeled as
TODOs instead of being fabricated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..utils.exceptions import AdaptShotError, BufferCapacityError, ConfigValidationError, InvalidImageError
from .utils import (
    SESSION_PATH,
    StudioSession,
    append_log,
    build_config,
    build_prediction_summary,
    build_report_markdown,
    build_support_summary,
    buffer_snapshot,
    choose_prediction_row,
    config_to_json,
    create_learner,
    collect_image_sources,
    estimate_buffer_memory_mb,
    estimate_runtime_profile,
    export_native_bundle,
    file_metadata_rows,
    gallery_items,
    infer_labels,
    load_session_snapshot,
    log_text,
    persist_session_snapshot,
    prediction_choices,
    prediction_rows,
    runtime_health_snapshot,
    save_log_file,
    summarize_labels,
    summarize_folder_imports,
    validate_file_bundle,
)

logger = logging.getLogger(__name__)

STUDIO_CSS = """
:root {
  --studio-bg: #f6f4ef;
  --studio-panel: rgba(255, 255, 255, 0.86);
  --studio-border: rgba(50, 50, 50, 0.08);
  --studio-ink: #1e1f24;
  --studio-accent: #1f6f78;
  --studio-accent-2: #c35c3a;
}

body {
  background:
    radial-gradient(circle at top left, rgba(31, 111, 120, 0.12), transparent 28%),
    radial-gradient(circle at bottom right, rgba(195, 92, 58, 0.10), transparent 30%),
    linear-gradient(180deg, #faf7f0 0%, #f2efe8 100%);
  color: var(--studio-ink);
}

.gradio-container {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

.studio-card {
  border: 1px solid var(--studio-border);
  border-radius: 18px;
  background: var(--studio-panel);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06);
  padding: 16px;
}

.studio-hero {
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(31, 111, 120, 0.95), rgba(16, 35, 42, 0.94));
  color: white;
  padding: 24px 24px 20px;
}

.studio-hero h1,
.studio-hero p {
  margin: 0;
}

.studio-muted {
  opacity: 0.8;
}

.studio-chip {
  display: inline-block;
  padding: 0.28rem 0.6rem;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  margin-right: 0.5rem;
  margin-top: 0.25rem;
}
"""


def _require_gradio() -> Any:
    """Import Gradio lazily and raise a friendly error if it is missing."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised only without gui deps
        raise RuntimeError(
            "AdaptShot Studio requires the optional gui extra. Install it with `pip install -e \".[gui]\"`."
        ) from exc
    return gr


def _coerce_file_paths(files: Optional[Sequence[Any]]) -> List[Path]:
    """Normalize Gradio file inputs into concrete file paths."""

    if not files:
        return []
    normalized: List[Path] = []
    for entry in files:
        raw_path = getattr(entry, "name", entry)
        normalized.append(Path(str(raw_path)).expanduser())
    return normalized


def _make_dataframe(records: Sequence[Mapping[str, Any]]) -> Any:
    """Build a pandas dataframe without importing it at module load time."""

    return list(records)


def _format_json(data: Mapping[str, Any]) -> str:
    """Pretty-print a mapping as JSON."""

    return json.dumps(data, indent=2, sort_keys=True)


def _unique_labels(labels: Sequence[Union[str, int]]) -> List[str]:
    """Return sorted unique labels for dropdown options."""

    unique = sorted({str(label) for label in labels})
    return unique


async def _to_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking function in a worker thread."""

    return await asyncio.to_thread(fn, *args, **kwargs)


class StudioController:
    """Stateful controller that keeps the studio callbacks small."""

    def __init__(self) -> None:
        """Initialize an empty studio controller."""

        self.session = load_session_snapshot(SESSION_PATH)
        if not self.session.logs:
            append_log(self.session, "Studio controller initialized.")

    def _store(self, session: StudioSession) -> StudioSession:
        """Store the current session and refresh diagnostics."""

        session.health_snapshot = runtime_health_snapshot()
        self.session = session
        return session

    def _config_values(
        self,
        backbone: str,
        eco_mode: bool,
        max_buffer_size: int,
        calibration_method: str,
        seed: int,
    ) -> Dict[str, Any]:
        """Collect config fields from the UI controls."""

        return {
            "backbone": backbone,
            "eco_mode": eco_mode,
            "max_buffer_size": max_buffer_size,
            "calibration_method": calibration_method,
            "seed": seed,
        }

    async def validate_config(
        self,
        backbone: str,
        eco_mode: bool,
        max_buffer_size: int,
        calibration_method: str,
        seed: int,
        session: StudioSession,
    ) -> Tuple[str, str, str, StudioSession]:
        """Validate configuration inputs and display heuristic estimates."""

        values = self._config_values(backbone, eco_mode, max_buffer_size, calibration_method, seed)
        try:
            config = build_config(values)
            profile = estimate_runtime_profile(config)
            preview = config_to_json(config)
            session.config_values = values
            session.last_error = ""
            append_log(session, "Configuration validated.")
            persist_session_snapshot(session, SESSION_PATH)
            self._store(session)
            message = "\n".join(
                [
                    "✅ Config validated.",
                    f"Heuristic idle RAM: {profile['idle_ram_mb']:.2f} MB",
                    f"Heuristic support RAM: {profile['support_ram_mb']:.2f} MB",
                    f"Heuristic inference latency: {profile['estimated_inference_ms']:.2f} ms",
                    "",
                    "These are estimates only; they are not benchmark claims.",
                ]
            )
            return preview, _format_json(profile), message, session
        except (ValueError, ConfigValidationError) as exc:
            session.last_error = str(exc)
            append_log(session, f"Configuration validation failed: {exc}", level="error")
            persist_session_snapshot(session, SESSION_PATH)
            self._store(session)
            return json.dumps(values, indent=2, sort_keys=True), "{}", f"❌ {exc}", session

    async def load_support_set(
        self,
        support_folder_text: str,
        support_folder_recursive: bool,
        support_files: Optional[Sequence[Any]],
        label_strategy: str,
        manual_labels: str,
        backbone: str,
        eco_mode: bool,
        max_buffer_size: int,
        calibration_method: str,
        seed: int,
        session: StudioSession,
    ) -> Tuple[str, str, Any, Any, str, StudioSession]:
        """Load support images into a fresh learner and summarize the class set."""

        paths = collect_image_sources(
            _coerce_file_paths(support_files),
            folder_text=support_folder_text,
            recursive=support_folder_recursive,
        )
        if not paths:
            message = "❌ No support images were provided. Upload files or enter a local folder path."
            session.last_error = message
            append_log(session, message, level="error")
            return message, "", [], [], "", session

        values = self._config_values(backbone, eco_mode, max_buffer_size, calibration_method, seed)
        try:
            normalized_paths = validate_file_bundle(paths)
            labels = infer_labels(normalized_paths, strategy=label_strategy, manual_labels=manual_labels)
            learner = create_learner(values)

            await _to_thread(
                learner.load_support_images,
                [str(path) for path in normalized_paths],
                [str(label) for label in labels],
            )

            session.learner = learner
            session.config_values = values
            session.support_paths = [str(path) for path in normalized_paths]
            session.support_labels = [str(label) for label in labels]
            session.support_rows = file_metadata_rows(normalized_paths, labels)
            session.buffer_rows, session.buffer_summary = buffer_snapshot(learner)
            session.query_paths = []
            session.prediction_rows = []
            session.prediction_choices = []
            session.selected_prediction_id = ""
            session.last_error = ""
            append_log(session, f"Loaded {len(normalized_paths)} support images.")
            persist_session_snapshot(session, SESSION_PATH)
            self._store(session)

            distribution = summarize_labels(labels)
            gallery = gallery_items(normalized_paths, labels)
            support_summary = build_support_summary(session.support_rows)
            folder_summary = summarize_folder_imports(normalized_paths)
            return (
                f"✅ Indexed {len(normalized_paths)} support images from {len(folder_summary)} folder(s).",
                support_summary,
                _make_dataframe(session.support_rows),
                gallery,
                _format_json(
                    {
                        "class_distribution": distribution,
                        "folder_distribution": folder_summary,
                        "embedding_count": len(getattr(learner, "_sim_embeddings", [])),
                    }
                ),
                session,
            )
        except (AdaptShotError, ValueError, ConfigValidationError, InvalidImageError) as exc:
            message = f"❌ {exc}"
            session.last_error = str(exc)
            append_log(session, message, level="error")
            persist_session_snapshot(session, SESSION_PATH)
            self._store(session)
            return message, "", [], [], "", session

    async def run_inference(
        self,
        query_folder_text: str,
        query_folder_recursive: bool,
        query_image: Optional[Any],
        query_files: Optional[Sequence[Any]],
        batch_mode: bool,
        session: StudioSession,
    ) -> Tuple[str, Any, Any, Any, str, str, StudioSession]:
        """Run inference on one image or a batch of query images."""

        learner = session.learner
        if learner is None:
            message = "❌ Load a support set before running inference."
            session.last_error = message
            append_log(session, message, level="error")
            return message, [], [], [], "", "", session

        paths = collect_image_sources(
            _coerce_file_paths(query_files),
            folder_text=query_folder_text,
            recursive=query_folder_recursive,
        )
        if batch_mode:
            if not paths:
                message = "❌ Batch mode is on, but no query images were provided. Upload files or enter a local folder path."
                session.last_error = message
                append_log(session, message, level="error")
                return message, [], [], [], "", "", session
        else:
            if query_image is not None:
                paths = _coerce_file_paths([query_image])
            if not paths:
                message = "❌ Upload a query image or provide a folder path first."
                session.last_error = message
                append_log(session, message, level="error")
                return message, [], [], [], "", "", session
            paths = [paths[0]]

        try:
            normalized_paths = validate_file_bundle(paths)
            results: List[Any] = []
            latencies_ms: List[float] = []
            for path in normalized_paths:
                start = time.perf_counter()
                result = await _to_thread(learner.predict, str(path))
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
                results.append(result)

            rows = prediction_rows(normalized_paths, results, latencies_ms)
            session.query_paths = [str(path) for path in normalized_paths]
            session.prediction_rows = rows
            session.prediction_choices = prediction_choices(rows)
            session.selected_prediction_id = session.prediction_rows[-1]["id"] if session.prediction_rows else ""
            session.buffer_rows, session.buffer_summary = buffer_snapshot(learner)
            session.ece_history.append(float(learner.calibrator.current_ece))
            session.last_error = ""
            append_log(session, f"Ran inference on {len(rows)} image(s).")
            persist_session_snapshot(session, SESSION_PATH)
            self._store(session)

            gallery = gallery_items(normalized_paths, [row["prediction"] for row in rows])
            summary = build_prediction_summary(rows)
            resource_log = f"Latency ms (mean): {sum(latencies_ms) / len(latencies_ms):.2f} | RSS: {runtime_health_snapshot()['rss_mb']:.2f} MB"
            return summary, _make_dataframe(rows), gallery, _format_json({"rows": rows}), resource_log, session.selected_prediction_id, session
        except (AdaptShotError, ValueError, ConfigValidationError, InvalidImageError) as exc:
            message = f"❌ {exc}"
            session.last_error = str(exc)
            append_log(session, message, level="error")
            persist_session_snapshot(session, SESSION_PATH)
            self._store(session)
            return message, [], [], [], "", "", session

    async def submit_correction(
        self,
        selected_prediction: str,
        suggested_label: str,
        custom_label: str,
        confidence_weight: float,
        session: StudioSession,
    ) -> Tuple[str, str, str, str, str, StudioSession]:
        """Submit a correction into the existing learner feedback router."""

        learner = session.learner
        if learner is None:
            message = "❌ Load a support set before submitting corrections."
            session.last_error = message
            append_log(session, message, level="error")
            return message, "", "", "", "", session

        chosen_row = choose_prediction_row(session.prediction_rows, selected_prediction)
        if chosen_row is None:
            message = "❌ Run inference before submitting a correction."
            session.last_error = message
            append_log(session, message, level="error")
            return message, "", "", "", "", session

        true_label = custom_label.strip() or suggested_label.strip()
        if not true_label:
            message = "❌ Provide a correction label before submitting."
            session.last_error = message
            append_log(session, message, level="error")
            return message, "", "", "", "", session

        try:
            result = await _to_thread(
                learner.correct,
                str(chosen_row["path"]),
                true_label,
                float(confidence_weight),
            )
            session.buffer_rows, session.buffer_summary = buffer_snapshot(learner)
            session.ece_history.append(float(learner.calibrator.current_ece))
            session.last_error = ""
            append_log(session, f"Correction routed for {chosen_row['file']} -> {true_label}.")
            persist_session_snapshot(session, SESSION_PATH)
            self._store(session)

            return (
                f"✅ Correction routed for {chosen_row['file']}.",
                str(result.get("buffer_size", "")),
                str(result.get("fine_tuned", False)),
                str(result.get("calibration_updated", False)),
                _format_json(result),
                session,
            )
        except (AdaptShotError, ValueError, ConfigValidationError, InvalidImageError) as exc:
            message = f"❌ {exc}"
            session.last_error = str(exc)
            append_log(session, message, level="error")
            persist_session_snapshot(session, SESSION_PATH)
            self._store(session)
            return message, "", "", "", "", session

    async def recalibrate(
        self,
        base_threshold: float,
        learning_rate: float,
        show_boundaries: bool,
        session: StudioSession,
    ) -> Tuple[str, str, Any, str, StudioSession]:
        """Recalibrate the learner and refresh ACT thresholds."""

        learner = session.learner
        if learner is None:
            message = "❌ Load a support set before recalibrating."
            session.last_error = message
            append_log(session, message, level="error")
            return message, "", [], "", session

        learner.act.eta = float(learning_rate)
        for class_idx in list(learner.act.get_all_thresholds().keys()):
            learner.act.reset_class(class_idx, base_threshold=float(base_threshold))

        if learner.calibrator.method == "temperature":
            min_samples = max(10, learner.calibrator.window_size // 2)
            if len(learner.calibrator._window_confidences) < min_samples:
                message = (
                    f"⚠️ Not enough observations to refit temperature yet. Need {min_samples}, "
                    f"have {len(learner.calibrator._window_confidences)}."
                )
                append_log(session, message, level="warning")
            else:
                await _to_thread(learner.calibrator._refit_temperature)
                message = "✅ Recalibration completed with temperature refit."
                append_log(session, message)
        else:
            message = "⚠️ Calibration method is conformal, so no temperature refit was applied."
            append_log(session, message, level="warning")

        thresholds = learner.act.get_all_thresholds()
        session.ece_history.append(float(learner.calibrator.current_ece))
        session.buffer_rows, session.buffer_summary = buffer_snapshot(learner)
        session.last_error = ""
        persist_session_snapshot(session, SESSION_PATH)
        self._store(session)

        boundary_text = "Decision boundaries shown." if show_boundaries else "Decision boundaries hidden."
        thresholds_preview = _format_json({str(key): round(value, 4) for key, value in list(thresholds.items())[:12]})
        plot_data = _make_dataframe([{"step": index + 1, "ece": value} for index, value in enumerate(session.ece_history[-100:])])
        return (
            f"{message} {boundary_text}",
            f"Temperature: {learner.calibrator.current_temperature:.4f} | ECE: {learner.calibrator.current_ece:.4f}",
            plot_data,
            thresholds_preview,
            session,
        )

    async def prune_buffer(self, pruning_enabled: bool, session: StudioSession) -> Tuple[str, Any, str, StudioSession]:
        """Prune the learner buffer using the existing capacity controls."""

        learner = session.learner
        if learner is None:
            message = "❌ Load a support set before pruning the buffer."
            session.last_error = message
            append_log(session, message, level="error")
            return message, [], "", session

        if not pruning_enabled:
            message = "ℹ️ Pruning is disabled. No changes were made."
            append_log(session, message)
            session.buffer_rows, session.buffer_summary = buffer_snapshot(learner)
            return message, _make_dataframe(session.buffer_rows), _format_json(session.buffer_summary), session

        before = len(getattr(learner, "_sim_embeddings", []))
        try:
            await _to_thread(learner._apply_buffer_management)
        except BufferCapacityError as exc:
            append_log(session, f"Buffer pruning used fallback behavior: {exc}", level="warning")

        session.buffer_rows, session.buffer_summary = buffer_snapshot(learner)
        after = len(getattr(learner, "_sim_embeddings", []))
        message = f"✅ Buffer pruned from {before} to {after} item(s)."
        if estimate_buffer_memory_mb(learner) > 200.0:
            message += " Warning: buffer memory is above 200 MB; consider loading fewer support examples or reducing max_buffer_size."
        session.last_error = ""
        append_log(session, message)
        persist_session_snapshot(session, SESSION_PATH)
        self._store(session)
        return message, _make_dataframe(session.buffer_rows), _format_json(session.buffer_summary), session

    async def export_model(self, export_format: str, session: StudioSession) -> Tuple[str, str, str, Optional[str], StudioSession]:
        """Export the current learner state to a zip package or TODO message."""

        learner = session.learner
        if learner is None:
            message = "❌ Load a support set before exporting."
            session.last_error = message
            append_log(session, message, level="error")
            return message, "", "", None, session

        if export_format != "native":
            message = "[TODO: Implement in src/adaptshot/core/learner.py] ONNX and TorchScript export are not available yet."
            append_log(session, message, level="warning")
            session.last_error = message
            return message, build_report_markdown(learner, session=session), "", None, session

        archive_path = await _to_thread(export_native_bundle, learner, session)
        report_text = build_report_markdown(learner, session=session)
        install_command = "pip install adaptshot"
        session.export_paths = [str(archive_path)]
        session.last_error = ""
        append_log(session, f"Exported native bundle to {archive_path}.")
        persist_session_snapshot(session, SESSION_PATH)
        self._store(session)
        return (
            f"✅ Exported native bundle to {archive_path.name}.",
            report_text,
            install_command,
            str(archive_path),
            session,
        )

    async def export_logs(self, session: StudioSession) -> Tuple[str, str, StudioSession]:
        """Export the current log buffer to a local file."""

        if not session.logs:
            message = "No log entries have been collected yet."
            return message, "", session

        log_path = await _to_thread(save_log_file, session.logs)
        append_log(session, f"Exported logs to {log_path}.")
        persist_session_snapshot(session, SESSION_PATH)
        self._store(session)
        return f"✅ Log file written to {log_path.name}.", str(log_path), session

    async def health_check(self, session: StudioSession) -> Tuple[str, str, str, StudioSession]:
        """Refresh local diagnostics for the diagnostics tab."""

        session.health_snapshot = runtime_health_snapshot()
        learner = session.learner
        health = runtime_health_snapshot()
        if learner is not None:
            health["support_size"] = len(getattr(learner, "_sim_embeddings", []))
            health["current_ece"] = round(float(learner.calibrator.current_ece), 6)
            health["current_temperature"] = round(float(learner.calibrator.current_temperature), 6)
        append_log(session, "Health check refreshed.")
        persist_session_snapshot(session, SESSION_PATH)
        self._store(session)
        return _format_json(health), build_support_summary(session.support_rows), log_text(session), session


def _studio_intro() -> str:
    """Return a compact hero section for the app."""

    return (
        "<div class='studio-hero'>"
        "<h1>AdaptShot Studio</h1>"
        "<p>Offline, CPU-first few-shot vision workflows for non-coders.</p>"
        "<div style='margin-top:10px;'>"
        "<span class='studio-chip'>Local only</span>"
        "<span class='studio-chip'>Optional gui extra</span>"
        "<span class='studio-chip'>Truthful metrics</span>"
        "</div>"
        "</div>"
    )


def build_ui() -> Any:
    """Build the Gradio interface for AdaptShot Studio."""

    gr = _require_gradio()
    controller = StudioController()
    session = controller.session

    with gr.Blocks(title="AdaptShot Studio", analytics_enabled=False) as demo:
        gr.HTML(_studio_intro())
        gr.Markdown(
            "This workspace wraps the existing AdaptShot learner. Where a backend feature does not exist yet, the UI shows a TODO instead of pretending it is implemented."
        )

        state = gr.State(session)

        with gr.Tab("1. Setup & Configuration"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    backbone = gr.Dropdown(
                        choices=["resnet18", "mobilenet_v3_small"],
                        value=session.config_values["backbone"],
                        label="Backbone",
                        info="Choose the lightweight feature extractor used by the learner.",
                    )
                    eco_mode = gr.Checkbox(
                        value=session.config_values["eco_mode"],
                        label="eco_mode",
                        info="When enabled, the app prefers lower resource use over responsiveness.",
                    )
                    max_buffer_size = gr.Slider(
                        minimum=10,
                        maximum=500,
                        step=1,
                        value=session.config_values["max_buffer_size"],
                        label="max_buffer_size",
                    )
                    calibration_method = gr.Dropdown(
                        choices=["temperature", "conformal"],
                        value=session.config_values["calibration_method"],
                        label="Calibration method",
                    )
                    seed = gr.Number(value=session.config_values["seed"], precision=0, label="seed")
                    validate_config_btn = gr.Button("Validate Config", variant="primary")

                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    config_preview = gr.Textbox(label="Config preview", lines=16, interactive=False)
                    runtime_profile = gr.Textbox(label="RAM / latency estimate", lines=8, interactive=False)
                    config_feedback = gr.Textbox(label="Validation result", lines=6, interactive=False)

            validate_config_btn.click(
                fn=controller.validate_config,
                inputs=[backbone, eco_mode, max_buffer_size, calibration_method, seed, state],
                outputs=[config_preview, runtime_profile, config_feedback, state],
            )

        with gr.Tab("2. Dataset & Class Management"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    support_folder_text = gr.Textbox(
                        label="Support folder path(s)",
                        placeholder="Example: /home/user/dataset/class_a, /home/user/dataset/class_b",
                        lines=2,
                    )
                    support_folder_recursive = gr.Checkbox(
                        value=True,
                        label="Scan support folders recursively",
                    )
                    support_files = gr.File(file_count="multiple", label="Support images", type="filepath")
                    label_strategy = gr.Dropdown(
                        choices=["folder", "stem", "manual"],
                        value="folder",
                        label="Label strategy",
                        info="Folder grouping uses the uploaded parent folder when available; otherwise choose stem or manual labels.",
                    )
                    manual_labels = gr.Textbox(
                        label="Manual labels",
                        lines=4,
                        placeholder="Enter one label per line, or one label to apply to all images.",
                    )
                    load_support_btn = gr.Button("Load Support Set", variant="primary")

                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    support_status = gr.Textbox(label="Load status", lines=3, interactive=False)
                    support_summary = gr.Textbox(label="Class distribution summary", lines=4, interactive=False)
                    embedding_summary = gr.Textbox(label="Embedding count and metadata", lines=6, interactive=False)
                    support_table = gr.Dataframe(label="Support table", interactive=False)
                    support_gallery = gr.Gallery(label="Support previews", columns=3, preview=True)

            load_support_btn.click(
                fn=controller.load_support_set,
                inputs=[support_folder_text, support_folder_recursive, support_files, label_strategy, manual_labels, backbone, eco_mode, max_buffer_size, calibration_method, seed, state],
                outputs=[support_status, support_summary, support_table, support_gallery, embedding_summary, state],
            )

        with gr.Tab("3. Train & Predict"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    query_folder_text = gr.Textbox(
                        label="Query folder path(s)",
                        placeholder="Example: /home/user/test_images or /home/user/test_folder_a, /home/user/test_folder_b",
                        lines=2,
                    )
                    query_folder_recursive = gr.Checkbox(
                        value=True,
                        label="Scan query folders recursively",
                    )
                    query_image = gr.Image(type="filepath", label="Query image")
                    query_files = gr.File(file_count="multiple", label="Batch query images", type="filepath")
                    batch_mode = gr.Checkbox(
                        value=False,
                        label="Batch mode",
                        info="When enabled, the app processes the uploaded file list one by one.",
                    )
                    run_inference_btn = gr.Button("Run Inference", variant="primary")

                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    inference_status = gr.Textbox(label="Inference status", lines=3, interactive=False)
                    prediction_table = gr.Dataframe(label="Predictions", interactive=False)
                    prediction_gallery = gr.Gallery(label="Prediction previews", columns=3, preview=True)
                    prediction_summary = gr.Textbox(label="Inference summary", lines=4, interactive=False)
                    latency_summary = gr.Textbox(label="Resource log", lines=4, interactive=False)
                    prediction_selector = gr.Dropdown(label="Select a prediction for correction", choices=[], allow_custom_value=True)

            run_inference_btn.click(
                fn=controller.run_inference,
                inputs=[query_folder_text, query_folder_recursive, query_image, query_files, batch_mode, state],
                outputs=[inference_status, prediction_table, prediction_gallery, prediction_summary, latency_summary, prediction_selector, state],
            )

        with gr.Tab("4. Human-in-the-Loop Correction"):
            gr.Markdown(
                "Select a prediction, enter the correct label, then submit the correction. The learner routes it through the existing feedback pipeline."
            )
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    correction_target = gr.Dropdown(label="Prediction to correct", choices=[], allow_custom_value=True)
                    suggested_label = gr.Dropdown(label="Suggested label", choices=[], allow_custom_value=True)
                    custom_label = gr.Textbox(
                        label="Custom label",
                        placeholder="Optional if the dropdown already contains the correct class name.",
                    )
                    human_confidence = gr.Slider(
                        0.0,
                        1.0,
                        value=1.0,
                        step=0.01,
                        label="Human confidence weight",
                        info="Higher values tell the learner you are very confident in the correction.",
                    )
                    correction_btn = gr.Button("Submit Correction", variant="primary")

                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    correction_status = gr.Textbox(label="Correction status", lines=3, interactive=False)
                    correction_buffer = gr.Textbox(label="Buffer size", lines=2, interactive=False)
                    correction_finetune = gr.Textbox(label="Fine-tune trigger status", lines=2, interactive=False)
                    correction_calibration = gr.Textbox(label="Calibration update", lines=2, interactive=False)
                    correction_payload = gr.Textbox(label="Routing payload", lines=8, interactive=False)

            correction_btn.click(
                fn=controller.submit_correction,
                inputs=[correction_target, suggested_label, custom_label, human_confidence, state],
                outputs=[correction_status, correction_buffer, correction_finetune, correction_calibration, correction_payload, state],
            )

        with gr.Tab("5. Calibration & ACT Tuning"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    base_threshold = gr.Slider(0.5, 0.95, value=0.65, step=0.01, label="base_threshold")
                    learning_rate = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="learning_rate")
                    show_boundaries = gr.Checkbox(value=True, label="Show ACT decision boundaries")
                    recalibrate_btn = gr.Button("Recalibrate", variant="primary")

                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    calibration_status = gr.Textbox(label="Calibration status", lines=3, interactive=False)
                    calibration_metrics = gr.Textbox(label="Current temperature and ECE", lines=3, interactive=False)
                    ece_plot = gr.Dataframe(label="ECE sliding window", interactive=False)
                    threshold_table = gr.Textbox(label="Thresholds by class", lines=8, interactive=False)

            recalibrate_btn.click(
                fn=controller.recalibrate,
                inputs=[base_threshold, learning_rate, show_boundaries, state],
                outputs=[calibration_status, calibration_metrics, ece_plot, threshold_table, state],
            )

        with gr.Tab("6. Buffer & Memory Management"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    pruning_enabled = gr.Checkbox(value=True, label="UP-UGF pruning enabled")
                    prune_btn = gr.Button("Prune Buffer", variant="primary")

                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    buffer_status = gr.Textbox(label="Buffer status", lines=3, interactive=False)
                    buffer_table = gr.Dataframe(label="Buffer contents", interactive=False)
                    buffer_summary = gr.Textbox(label="Memory summary", lines=4, interactive=False)

            prune_btn.click(
                fn=controller.prune_buffer,
                inputs=[pruning_enabled, state],
                outputs=[buffer_status, buffer_table, buffer_summary, state],
            )

        with gr.Tab("7. Export & Deployment"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    export_format = gr.Dropdown(
                        choices=["native", "torchscript", "onnx stub"],
                        value="native",
                        label="Export format",
                    )
                    export_btn = gr.Button("Export Package", variant="primary")
                    copy_install_btn = gr.Button("Copy Install Command")

                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    export_status = gr.Textbox(label="Export status", lines=3, interactive=False)
                    report_preview = gr.Textbox(label="Export report", lines=12, interactive=False)
                    install_command = gr.Textbox(label="Install command", lines=2, interactive=False)
                    archive_file = gr.File(label="Download archive")

            export_btn.click(
                fn=controller.export_model,
                inputs=[export_format, state],
                outputs=[export_status, report_preview, install_command, archive_file, state],
            )

            def _copy_install_command() -> str:
                """Return the install command for sharing."""

                return "pip install adaptshot"

            copy_install_btn.click(fn=_copy_install_command, inputs=[], outputs=[install_command])

        with gr.Tab("8. Logs & Diagnostics"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    refresh_health_btn = gr.Button("Refresh Health Check", variant="primary")
                    export_log_btn = gr.Button("Export Log")
                    health_snapshot = gr.Textbox(label="Health snapshot", lines=10, interactive=False)
                    diagnostic_summary = gr.Textbox(label="Diagnostic summary", lines=5, interactive=False)

                with gr.Column(scale=1, elem_classes=["studio-card"]):
                    log_console = gr.Textbox(label="Real-time console log", lines=18, interactive=False)
                    log_export = gr.File(label="Exported log")
                    gr.Markdown(
                        """
                        ### Troubleshooting
                        - If the app asks for support images first, load a support set before inference.
                        - If calibration says the window is not ready, collect more predictions or corrections.
                        - If export reports TODO for ONNX or TorchScript, use native export for now.
                        - If you need API details, consult [src/adaptshot/core/learner.py](src/adaptshot/core/learner.py).
                        """
                    )

            refresh_health_btn.click(
                fn=controller.health_check,
                inputs=[state],
                outputs=[health_snapshot, diagnostic_summary, log_console, state],
            )
            export_log_btn.click(
                fn=controller.export_logs,
                inputs=[state],
                outputs=[diagnostic_summary, log_export, state],
            )

        demo.load(
            fn=controller.health_check,
            inputs=[state],
            outputs=[health_snapshot, diagnostic_summary, log_console, state],
        )

    return demo


def launch(server_name: str = "127.0.0.1", server_port: int = 7860, inbrowser: bool = True) -> None:
    """Launch AdaptShot Studio locally."""

    gr = _require_gradio()
    demo = build_ui()
    ports_to_try = [server_port] + [port for port in range(server_port + 1, server_port + 6)]
    launch_error: Optional[Exception] = None

    for port in ports_to_try:
        try:
            demo.launch(
                server_name=server_name,
                server_port=port,
                inbrowser=inbrowser,
                share=False,
                css=STUDIO_CSS,
                theme=gr.themes.Base(),
            )
            return
        except OSError as exc:
            error_text = str(exc)
            if "Cannot find empty port" not in error_text and "Address already in use" not in error_text:
                raise
            launch_error = exc

    raise RuntimeError(
        f"Unable to launch AdaptShot Studio on ports {ports_to_try[0]}-{ports_to_try[-1]}. "
        "Please free the port or pass a different server_port."
    ) from launch_error


if __name__ == "__main__":
    launch()
