"""MziziGuard Web Application — Gradio-powered crop disease detection.

Run::

    python -m examples.mziziguard.app

Then open http://localhost:7860 in your browser.

Tabs:
  - **Setup** — Load support images (generate samples or upload real photos)
  - **Diagnose** — Upload a crop photo and get instant diagnosis
  - **Teach** — Correct wrong predictions (human-in-the-loop)
  - **Health** — View system calibration and session metrics
  - **Batch** — Process multiple images at once and export results
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple, cast

# Make AdaptShot importable from this script's context
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Lazy Gradio import (only installed with [ui] extras)
# ---------------------------------------------------------------------------
try:
    import gradio as gr  # type: ignore[import-not-found]
except ImportError:
    raise ImportError(
        "Gradio is not installed. Run: pip install adaptshot[ui]"
    ) from None

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from examples.mziziguard.engine import MziziGuard  # noqa: E402

# ---------------------------------------------------------------------------
# Globals (per-process, single-user Gradio app)
# ---------------------------------------------------------------------------
_guard: Optional[MziziGuard] = None
_config_path: Optional[str] = None


def _get_guard() -> MziziGuard:
    """Lazy-initialize the MziziGuard singleton."""
    global _guard
    if _guard is None:
        _guard = MziziGuard(config_path=_config_path)
    return _guard


# ===================================================================
# Tab 1: Setup — Load support images
# ===================================================================


def _setup_with_samples(n_support: int) -> str:
    """Generate synthetic samples and train the model."""
    try:
        guard = _get_guard()
        count = guard.initialize_with_samples(n_support=n_support)
        labels = guard.known_labels
        return (
            f"✅ Generated {count} support images across {len(labels)} classes:\n"
            + "\n".join(f"  • {label}" for label in labels)
            + "\n\n🧠 Model ready. Switch to the **Diagnose** tab to test."
        )
    except Exception as exc:
        return f"❌ Setup failed: {exc}"


def _setup_from_folder(folder_path: str, max_per_class: int) -> str:
    """Load real images from a folder structure."""
    if not folder_path or not os.path.isdir(folder_path):
        return "❌ Please select a valid folder with class subdirectories."
    try:
        guard = _get_guard()
        count = guard.load_images_from_dir(folder_path, max_per_class=max_per_class)
        labels = guard.known_labels
        return (
            f"✅ Loaded {count} images from folder.\n"
            + f"Detected {len(labels)} classes:\n"
            + "\n".join(f"  • {label}" for label in labels)
            + "\n\n🧠 Model ready. Switch to the **Diagnose** tab."
        )
    except Exception as exc:
        return f"❌ Loading failed: {exc}"


def _setup_status() -> str:
    """Return current model status."""
    guard = _get_guard()
    if guard.is_trained:
        labels = guard.known_labels
        return (
            f"🟢 **Trained** — {len(labels)} classes loaded.\n"
            + "\n".join(f"  • {label}" for label in labels)
        )
    return "⚪ **Not trained** — Generate samples or load images first."


# ===================================================================
# Tab 2: Diagnose — Predict disease
# ===================================================================


def _diagnose(image: Optional[str]) -> Tuple[str, str, float, str, str]:
    """Run diagnosis on an uploaded image."""
    if image is None:
        return "", "", 0.0, "", "⚠️ Upload an image first."

    guard = _get_guard()
    if not guard.is_trained:
        return "", "", 0.0, "", "❌ Model not trained. Go to Setup tab first."

    try:
        result = guard.diagnose(image)

        # Build markdown summary
        severity_emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}.get(
            result.severity, "⚪"
        )

        summary = (
            f"## {severity_emoji} Diagnosis: {result.swahili}\n\n"
            f"**English:** {result.label}\n\n"
            f"**Confidence:** {result.confidence:.1%}\n\n"
            f"**Severity:** {result.severity.upper()}\n\n"
        )

        action = result.action
        confidence_pct = result.confidence
        detail = (
            f"ACT Decision: {result.act_action}\n"
            f"OOD Flag: {result.ood_flag}\n"
            f"Distance: {result.distance_to_prototype:.4f}"
        )

        if result.ood_flag:
            warning = "🚫 **OUT-OF-DISTRIBUTION** — This doesn't look like any known crop. The model is being honest: it doesn't know."
        elif result.uncertainty_flag:
            warning = "⚠️ **Uncertain** — The model isn't confident. A human should verify this."
        else:
            warning = "✅ Confident prediction."

        return summary, action, confidence_pct, detail, warning

    except Exception as exc:
        return "", "", 0.0, "", f"❌ Error: {exc}"


# ===================================================================
# Tab 3: Teach — Human correction
# ===================================================================


def _get_label_choices() -> List[str]:
    guard = _get_guard()
    if guard.is_trained:
        return guard.known_labels
    return ["healthy_maize", "northern_leaf_blight", "gray_leaf_spot"]


def _teach(true_label: str, confidence_weight: float) -> str:
    """Submit a human correction."""
    guard = _get_guard()
    if not guard.is_trained:
        return "❌ Model not trained yet."
    if not true_label:
        return "❌ Select the correct label."

    return guard.teach_from_ui(
        true_label=true_label,
        confidence_weight=confidence_weight,
    )


# ===================================================================
# Tab 4: Health — System dashboard
# ===================================================================


def _health_report() -> str:
    """Return human-readable health report."""
    guard = _get_guard()
    if not guard.is_trained:
        return "❌ Model not trained yet. No health data available."

    health = guard.system_health()
    calib = health.get("calibration", {})
    session = health.get("session", {})
    config = health.get("config", {})

    return (
        f"## 🏥 System Health\n\n"
        f"**Status:** {health.get('status', 'unknown')}\n\n"
        f"---\n"
        f"### 📊 Calibration\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| ECE | {calib.get('ece', 'N/A')} |\n"
        f"| Debiased ECE | {calib.get('debiased_ece', 'N/A')} |\n"
        f"| Temperature | {calib.get('temperature', 'N/A')} |\n"
        f"| Window Size | {calib.get('window_size', 'N/A')} |\n"
        f"| OOD Threshold | {calib.get('ood_distance_threshold', 'N/A')} |\n"
        f"| Support Size | {calib.get('support_size', 'N/A')} |\n"
        f"| Prototype Count | {calib.get('prototype_count', 'N/A')} |\n\n"
        f"---\n"
        f"### 📈 Session\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total Predictions | {session.get('total_predictions', 0)} |\n"
        f"| Total Corrections | {session.get('total_corrections', 0)} |\n"
        f"| Accuracy | {session.get('accuracy', 1.0)} |\n"
        f"| Session Duration | {session.get('session_duration_seconds', 0)}s |\n\n"
        f"---\n"
        f"### ⚙️ Config\n"
        f"| Setting | Value |\n"
        f"|---------|-------|\n"
        f"| Backbone | {config.get('backbone', 'N/A')} |\n"
        f"| Device | {config.get('device', 'N/A')} |\n"
        f"| Eco Mode | {config.get('eco_mode', 'N/A')} |\n"
        f"| Known Classes | {config.get('known_classes', 0)} |\n"
    )


# ===================================================================
# Tab 5: Batch — Process multiple images
# ===================================================================


def _batch_diagnose(files: List[str]) -> str:
    """Process a batch of images."""
    if not files:
        return "⚠️ Upload images to process."

    guard = _get_guard()
    if not guard.is_trained:
        return "❌ Model not trained yet."

    paths = [str(f.name if hasattr(f, "name") else f) for f in files]
    results = guard.batch_diagnose(paths)

    # Build markdown table
    lines = [
        "| # | Image | Diagnosis (Swahili) | Confidence | Severity | Action |",
        "|---|-------|---------------------|------------|----------|--------|",
    ]
    for i, (path, r) in enumerate(zip(paths, results), 1):
        fname = os.path.basename(path)[:30]
        sev = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴", "unknown": "⚪"}.get(
            r.severity, "⚪"
        )
        lines.append(
            f"| {i} | {fname} | {r.swahili} | {r.confidence:.1%} | {sev} {r.severity} | {r.action[:60]}... |"
        )

    return "\n".join(lines)


# ===================================================================
# Build the UI
# ===================================================================

_CSS = """
.severity-low { color: #22c55e; }
.severity-moderate { color: #eab308; }
.severity-high { color: #f97316; }
.severity-critical { color: #ef4444; }
footer { visibility: hidden; }
"""


def build_app() -> gr.Blocks:
    """Construct the full MziziGuard Gradio application."""

    with gr.Blocks(
        title="MziziGuard — Crop Disease Detection",
    ) as app:
        # ── Header ──
        gr.Markdown(
            """
            # 🌽 MziziGuard — Crop Disease Detection
            **Powered by AdaptShot v0.1.1** · CPU-only · Offline · Few-Shot Learning

            *Mlinzi wa mazao kwa wakulima wadogo — Crop guardian for smallholder farmers.*
            """
        )

        # ── Tab bar ──
        with gr.Tabs():
            # ========== TAB 1: SETUP ==========
            with gr.TabItem("⚙️ Setup"):
                gr.Markdown(
                    """
                    ### Train the model with support images

                    **Option A:** Generate synthetic sample images (quick, zero data needed).
                    **Option B:** Upload your own real crop photos organized in folders.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Option A: Synthetic Samples")
                        n_support = gr.Slider(
                            1, 20, value=5, step=1,
                            label="Images per class",
                            info="More images = better accuracy.",
                        )
                        gen_btn = gr.Button("🎲 Generate Samples & Train", variant="primary")
                        gen_status = gr.Textbox(
                            label="Result", interactive=False, lines=6,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### Option B: Real Images")
                        folder_input = gr.Textbox(
                            label="Image folder path",
                            placeholder="/path/to/images/",
                            info="Folder should have one subfolder per disease class.",
                        )
                        max_per = gr.Number(
                            value=0, label="Max images per class (0 = unlimited)",
                            precision=0,
                        )
                        folder_btn = gr.Button("📂 Load from Folder", variant="primary")
                        folder_status = gr.Textbox(
                            label="Result", interactive=False, lines=6,
                        )

                gr.Markdown("---")
                gr.Markdown("#### Current Model Status")
                status_btn = gr.Button("🔍 Check Status")
                status_text = gr.Markdown(
                    "⚪ Not trained yet. Generate samples or load images above."
                )

                # Wiring
                gen_btn.click(
                    fn=_setup_with_samples,
                    inputs=[n_support],
                    outputs=[gen_status],
                )
                folder_btn.click(
                    fn=_setup_from_folder,
                    inputs=[folder_input, max_per],
                    outputs=[folder_status],
                )
                status_btn.click(
                    fn=_setup_status,
                    inputs=[],
                    outputs=[status_text],
                )

            # ========== TAB 2: DIAGNOSE ==========
            with gr.TabItem("🔍 Diagnose"):
                gr.Markdown(
                    """
                    ### Upload a crop photo for instant diagnosis

                    Take a photo of a diseased leaf and let MziziGuard identify it.
                    Works **without internet** — just like in the field.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        query_image = gr.Image(
                            type="filepath", label="Crop Photo",
                            height=300,
                        )
                        predict_btn = gr.Button(
                            "🔬 Diagnose", variant="primary", size="lg",
                        )

                    with gr.Column(scale=1):
                        diagnosis_md = gr.Markdown(
                            "Upload a photo and click **Diagnose**."
                        )
                        confidence_bar = gr.Label(
                            label="Confidence", value={},
                        )
                        action_text = gr.Textbox(
                            label="📋 Recommended Action", interactive=False,
                            lines=3,
                        )

                with gr.Accordion("Technical Details", open=False):
                    detail_text = gr.Textbox(
                        label="Raw prediction data", interactive=False,
                    )

                warning_box = gr.Markdown("")

                # Wiring
                predict_btn.click(
                    fn=_diagnose,
                    inputs=[query_image],
                    outputs=[
                        diagnosis_md, action_text, confidence_bar,
                        detail_text, warning_box,
                    ],
                )

            # ========== TAB 3: TEACH ==========
            with gr.TabItem("👩‍🏫 Teach"):
                gr.Markdown(
                    """
                    ### Correct wrong predictions

                    This is the **most powerful feature** of AdaptShot. When the model
                    gets something wrong, you can correct it — and it learns immediately.

                    **How it works:**
                    1. Go to the **Diagnose** tab and make a prediction.
                    2. If it's wrong, come here and tell it the correct label.
                    3. The model updates instantly — no retraining needed.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        true_label = gr.Dropdown(
                            label="Correct Label",
                            choices=_get_label_choices(),
                            interactive=True,
                            info="What disease is this actually?",
                            allow_custom_value=True,
                        )
                        conf_weight = gr.Slider(
                            0.0, 1.0, value=1.0, step=0.1,
                            label="Your Confidence",
                            info="How sure are you? (1.0 = completely sure)",
                        )
                        teach_btn = gr.Button(
                            "✅ Submit Correction", variant="primary",
                        )

                    with gr.Column(scale=1):
                        teach_status = gr.Textbox(
                            label="Correction Result", interactive=False,
                            lines=3,
                        )
                        gr.Markdown(
                            """
                            💡 **Tip:** The more corrections you provide,
                            the smarter MziziGuard becomes. Every correction
                            helps the next farmer get a better diagnosis.
                            """
                        )

                refresh_btn = gr.Button("🔄 Refresh Label List")
                refresh_btn.click(
                    fn=_get_label_choices,
                    inputs=[],
                    outputs=[true_label],
                )

                teach_btn.click(
                    fn=_teach,
                    inputs=[true_label, conf_weight],
                    outputs=[teach_status],
                )

            # ========== TAB 4: HEALTH ==========
            with gr.TabItem("🏥 System Health"):
                gr.Markdown(
                    """
                    ### System calibration & performance dashboard

                    Monitor how well MziziGuard is performing. Check calibration
                    error, temperature, and session statistics.
                    """
                )
                health_btn = gr.Button("🩺 Run Health Check", variant="primary")
                health_md = gr.Markdown("Click **Run Health Check** to see the report.")

                health_btn.click(
                    fn=_health_report,
                    inputs=[],
                    outputs=[health_md],
                )

            # ========== TAB 5: BATCH ==========
            with gr.TabItem("📦 Batch"):
                gr.Markdown(
                    """
                    ### Process multiple images at once

                    Upload multiple images and get a summary table of diagnoses.
                    Useful for extension officers processing photos from many farmers.
                    """
                )
                batch_files = gr.Files(
                    file_count="multiple",
                    label="Upload Crop Photos",
                    file_types=["image"],
                )
                batch_btn = gr.Button("📊 Batch Diagnose", variant="primary")
                batch_result = gr.Markdown("Upload images and click **Batch Diagnose**.")

                batch_btn.click(
                    fn=_batch_diagnose,
                    inputs=[batch_files],
                    outputs=[batch_result],
                )

        # ── Footer ──
        gr.Markdown(
            """
            ---
            **MziziGuard** · Powered by [AdaptShot](https://github.com/johnson2006christopher/adaptshot)
            · MIT License · Runs offline on CPU · Built for Tanzania 🇹🇿
            """
        )

    return cast(gr.Blocks, app)


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MziziGuard — Crop Disease Detection powered by AdaptShot"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for the Gradio server (default: 7860)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable link",
    )
    args = parser.parse_args()

    _config_path = args.config
    _guard = MziziGuard(config_path=_config_path)

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=_CSS,
    )
