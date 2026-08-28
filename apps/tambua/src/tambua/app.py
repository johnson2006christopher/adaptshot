"""Tambua web application — a Gradio interface over whatever the config describes.

Run::

    python -m examples.mziziengine.app

Then open http://localhost:7860 in your browser.

Tabs:
  - **Setup** — Load support images (generate samples or upload real photos)
  - **Diagnose** — Upload a crop photo and get instant diagnosis
  - **Teach** — Correct wrong predictions (human-in-the-loop)
  - **Health** — View system calibration and session metrics
  - **Batch** — Process multiple images at once and export results
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import cast

# No sys.path manipulation here. This is an installed package that declares
# `adaptshot` as a dependency, so the import resolves the way it does for any
# other user of the library. The previous version inserted the repository root
# into sys.path, which only worked when launched from one specific directory --
# the same defect #11 removed from the library itself.

# Gradio callbacks are a UI boundary: an exception escaping one takes down the tab
# instead of telling the user anything, so they catch broadly on purpose. What they
# must not do is swallow the traceback -- a user reporting "Setup failed" then leaves
# the maintainer nothing to work with. Every broad handler here logs before it
# returns its friendly message.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gradio is a hard dependency of this application (see pyproject.toml), not an
# optional extra of the library. Reaching this branch means the install is broken
# or was made with --no-deps, so name the package that actually provides it.
# ---------------------------------------------------------------------------
try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is not installed, but Tambua requires it. Reinstall the "
        "application with: pip install tambua"
    ) from None

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
import adaptshot
from tambua import data
from tambua.config import load_config
from tambua.engine import (
    DEFAULT_CONFIG,
    Identification,
    TambuaEngine,
    bundled_config,
    bundled_configs,
)

# ---------------------------------------------------------------------------
# Globals (per-process, single-user Gradio app)
# ---------------------------------------------------------------------------
_engine: TambuaEngine | None = None
_config_path: str | None = None


def _get_engine() -> TambuaEngine:
    """Lazy-initialise the engine singleton."""
    global _engine
    if _engine is None:
        _engine = TambuaEngine(config_path=_config_path)
    return _engine


# ===================================================================
# Tab 1: Setup — Load support images
# ===================================================================


def _check_folder(folder_path: str) -> str:
    """Report whether a folder can support training, without training on it.

    Separate from loading so that someone assembling a dataset can iterate --
    add photographs, check again -- instead of discovering the shortfall as a
    failure at the end of a training run.
    """

    if not folder_path:
        return "Enter the path to a folder of photographs."
    engine = _get_engine()
    problems = data.inspect_folder(folder_path, engine.cfg.labels)
    if not problems:
        return "✅ This folder is usable. Click **Load & train**."
    return "⚠️ " + data.render_problems(problems)


def _setup_from_folder(folder_path: str, max_per_class: int) -> str:
    """Load real images from a folder structure."""
    if not folder_path or not os.path.isdir(folder_path):
        return "❌ Please select a valid folder with class subdirectories."
    try:
        engine = _get_engine()
        count = engine.load_images_from_dir(folder_path, max_per_class=max_per_class)
        labels = engine.known_labels
        return (
            f"✅ Loaded {count} images from folder.\n"
            + f"Detected {len(labels)} classes:\n"
            + "\n".join(f"  • {label}" for label in labels)
            + "\n\n🧠 Model ready. Switch to the **Diagnose** tab."
        )
    except Exception as exc:
        logger.exception("folder load failed")
        return f"❌ Loading failed: {exc}"


def _setup_status() -> str:
    """Return current model status."""
    engine = _get_engine()
    if engine.is_trained:
        labels = engine.known_labels
        return (
            f"🟢 **Trained** — {len(labels)} classes loaded.\n"
            + "\n".join(f"  • {label}" for label in labels)
        )
    return "⚪ **Not trained** — Generate samples or load images first."


# ===================================================================
# Tab 2: Diagnose — Predict disease
# ===================================================================


def _render_prediction_set(
    engine: TambuaEngine, result: Identification, severity_emoji: str
) -> str:
    """Present the conformal prediction set as a set, not a ranked list.

    A single label with a percentage is a claim about one answer. A conformal set
    is a claim about a set, and unlike the percentage it is a claim that can be
    checked -- which is the entire reason to build on AdaptShot rather than on a
    plain classifier.

    So the set leads. When it holds more than one class the interface says "one
    of these", never a winner with the alternatives in small print: a person
    reading a top-1 has already stopped reading by the time the caveat arrives.
    """

    if result.is_abstention:
        return (
            "## 🤷 Not confident enough to name it\n\n"
            "Nothing was plausible enough to include. This needs someone who "
            "can look at it directly."
        )

    members = engine.set_members(result)

    if len(members) == 1:
        head = f"## {severity_emoji} {members[0].local_name}\n\n"
        head += f"**Label:** {members[0].key}\n\n"
    else:
        names = " **or** ".join(member.local_name for member in members)
        head = f"## 🤔 One of these {len(members)}: {names}\n\n"
        head += "\n".join(
            f"- **{member.local_name}** (`{member.key}`) — {member.severity}"
            for member in members
        )
        head += "\n\n"

    if result.coverage_is_measured:
        head += (
            f"The right answer falls inside this set about "
            f"**{result.empirical_coverage:.0%}** of the time, measured over "
            f"{result.calibration_size} calibration scores "
            f"(target {1 - result.alpha:.0%}).\n\n"
        )
    else:
        # Reporting `1 - alpha` here would be quoting the target as if it were a
        # measurement, which is the mistake #17 existed to correct.
        head += (
            f"⚠️ **Not yet calibrated.** Only {result.calibration_size} "
            "calibration scores so far, so this is the model's top guess rather "
            "than a set with a coverage guarantee. Correct a few predictions in "
            "the **Teach** tab and the guarantee becomes measurable.\n\n"
        )

    head += f"**Top-1 confidence:** {result.confidence:.1%}\n\n"
    return head


def _identify(image: str | None) -> tuple[str, str, float, str, str]:
    """Run diagnosis on an uploaded image."""
    if image is None:
        return "", "", 0.0, "", "⚠️ Upload an image first."

    engine = _get_engine()
    if not engine.is_trained:
        return "", "", 0.0, "", "❌ Model not trained. Go to Setup tab first."

    try:
        result = engine.identify(image)

        # Build markdown summary
        severity_emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}.get(
            result.severity, "⚪"
        )

        summary = _render_prediction_set(engine, result, severity_emoji)
        action = engine.advice_for(result)
        confidence_pct = result.confidence
        detail = (
            f"ACT Decision: {result.act_action}\n"
            f"OOD Flag: {result.ood_flag}\n"
            f"Distance: {result.distance_to_prototype:.4f}\n"
            f"Set size: {len(result.prediction_set)} at alpha={result.alpha:.2f}\n"
            f"Calibration scores: {result.calibration_size}"
        )

        if result.ood_flag:
            warning = "🚫 **OUT-OF-DISTRIBUTION** — This doesn't look like any known crop. The model is being honest: it doesn't know."
        elif result.uncertainty_flag:
            warning = "⚠️ **Uncertain** — The model isn't confident. A human should verify this."
        else:
            warning = "✅ Confident prediction."

        return summary, action, confidence_pct, detail, warning

    except Exception as exc:
        logger.exception("identify failed")
        return "", "", 0.0, "", f"❌ Error: {exc}"


# ===================================================================
# Tab 3: Teach — Human correction
# ===================================================================


def _get_label_choices() -> list[str]:
    """Labels offered in the correction dropdown.

    Before training there are no learned labels, but the config always describes
    the full set -- so the fallback is the configured vocabulary, not a fixed
    list. The previous fallback named three maize diseases, which meant a
    technician running the solar configuration was offered crop diseases to
    correct a solar panel to.
    """

    engine = _get_engine()
    if engine.is_trained:
        return engine.known_labels
    return engine.cfg.labels


def _teach(true_label: str, confidence_weight: float) -> str:
    """Submit a human correction."""
    engine = _get_engine()
    if not engine.is_trained:
        return "❌ Model not trained yet."
    if not true_label:
        return "❌ Select the correct label."

    return engine.teach_from_ui(
        true_label=true_label,
        confidence_weight=confidence_weight,
    )


# ===================================================================
# Tab 4: Health — System dashboard
# ===================================================================


def _health_report() -> str:
    """Return human-readable health report."""
    engine = _get_engine()
    if not engine.is_trained:
        return "❌ Model not trained yet. No health data available."

    health = engine.system_health()
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


def _batch_identify(files: list[str]) -> str:
    """Process a batch of images."""
    if not files:
        return "⚠️ Upload images to process."

    engine = _get_engine()
    if not engine.is_trained:
        return "❌ Model not trained yet."

    paths = [str(f.name if hasattr(f, "name") else f) for f in files]
    results = engine.batch_identify(paths)

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
            f"| {i} | {fname} | {r.local_name} | {r.confidence:.1%} | {sev} {r.severity} | {r.action[:60]}... |"
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
    """Construct the Gradio application for the loaded configuration.

    Every user-visible name comes from the config, so the same code renders
    MziziGuard or SolarCheck depending only on which file was loaded.
    """

    cfg = _get_engine().cfg
    app_name = cfg.application.name
    adaptshot_version = adaptshot.__version__
    _layout_example = data.describe_expected_layout(cfg.labels)
    domains = ", ".join(cfg.domains)

    with gr.Blocks(title=app_name) as app:
        # ── Header ──
        gr.Markdown(
            f"""
            # {app_name}
            {cfg.application.description}

            **AdaptShot v{adaptshot_version}** · CPU-only · Offline · Few-shot learning
            · Recognising: *{domains}*
            """
        )

        # ── Tab bar ──
        with gr.Tabs():
            # ========== TAB 1: SETUP ==========
            with gr.TabItem("⚙️ Setup"):
                gr.Markdown(
                    f"""
                    ### Teach the model with your own photographs

                    {app_name} ships no images. Five photographs per class is
                    enough to start — that is the whole point of few-shot
                    learning, and photographs you took yourself beat any
                    dataset for the thing you actually need to recognise.

                    Arrange them one folder per class, named exactly as in the
                    configuration:

                    ```
                    {_layout_example}
                    ```

                    **Check** tells you whether the folder is usable before you
                    spend a training run finding out.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        folder_input = gr.Textbox(
                            label="Photo folder path",
                            placeholder="/path/to/your_photos/",
                            info="One subfolder per class, named as in the config.",
                        )
                        max_per = gr.Number(
                            value=0, label="Max images per class (0 = unlimited)",
                            precision=0,
                        )
                        with gr.Row():
                            check_btn = gr.Button("🔍 Check folder")
                            folder_btn = gr.Button("📂 Load & train", variant="primary")

                    with gr.Column(scale=1):
                        folder_status = gr.Textbox(
                            label="Result", interactive=False, lines=12,
                        )

                gr.Markdown("---")
                gr.Markdown("#### Current Model Status")
                status_btn = gr.Button("🔍 Check Status")
                status_text = gr.Markdown(
                    "⚪ Not trained yet. Load photographs above."
                )

                # Wiring
                check_btn.click(
                    fn=_check_folder,
                    inputs=[folder_input],
                    outputs=[folder_status],
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
                    ### Upload a photo to identify it

                    Photograph the subject and let the model name it.
                    Works **without internet** — just as it will in the field.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        query_image = gr.Image(
                            type="filepath", label="Photo",
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
                    fn=_identify,
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
                            💡 **Tip:** Every correction you make teaches the
                            model. The next person to use it gets the benefit.
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

                    Monitor how well the model is performing. Check calibration
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

                    Upload multiple images and get one summary table.
                    Useful when working through a backlog of photographs.
                    """
                )
                batch_files = gr.Files(
                    file_count="multiple",
                    label="Upload photos",
                    file_types=["image"],
                )
                batch_btn = gr.Button("📊 Identify all", variant="primary")
                batch_result = gr.Markdown("Upload images and click **Identify all**.")

                batch_btn.click(
                    fn=_batch_identify,
                    inputs=[batch_files],
                    outputs=[batch_result],
                )

        # ── Footer ──
        gr.Markdown(
            f"""
            ---
            **{app_name}** · Powered by
            [AdaptShot](https://github.com/johnson2006christopher/adaptshot)
            · MIT License · Runs offline on CPU · Built in Tanzania 🇹🇿
            """
        )

    return cast(gr.Blocks, app)


# ===================================================================
# Entry point
# ===================================================================

def launch(argv: list[str] | None = None) -> None:
    """Console-script entry point for ``tambua``.

    Args:
        argv: Command-line arguments. ``None`` reads them from ``sys.argv``.
    """

    global _config_path, _engine

    parser = argparse.ArgumentParser(
        prog="tambua",
        description="Tambua — few-shot image classification, powered by AdaptShot",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to a domain config. Defaults to the bundled "
            f"{DEFAULT_CONFIG!r} configuration; run with --list-configs to see "
            "what else ships."
        ),
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List the configurations bundled with this installation, and exit.",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for the Gradio server (default: 7860)",
    )
    parser.add_argument(
        # Previously hardcoded to 0.0.0.0, which serves the UI to every machine on
        # the network the moment the app starts. That is a reasonable thing to ask
        # for -- a phone reaching a laptop over shared wifi -- but not a reasonable
        # default for a `pip install`-able app that accepts file uploads and writes
        # model files. Opt in explicitly.
        "--host", type=str, default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1, this machine only). "
             "Pass 0.0.0.0 to serve other devices on your network.",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable link",
    )
    args = parser.parse_args(argv)

    if args.list_configs:
        for name in bundled_configs():
            cfg = load_config(bundled_config(name))
            print(f"{name:16} {cfg.application.name} — {', '.join(cfg.domains)}")
        return

    _config_path = args.config
    _engine = TambuaEngine(config_path=_config_path)

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=_CSS,
    )


if __name__ == "__main__":
    launch()
