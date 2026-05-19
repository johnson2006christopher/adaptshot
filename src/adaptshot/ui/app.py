"""Gradio-based Human-in-the-Loop Interface for AdaptShot.

Provides a production-ready dashboard for:
1. Loading support sets
2. Predicting with calibrated uncertainty
3. Routing human corrections
4. Visualizing calibration and buffer state
"""

import os
import tempfile
from typing import List, Optional, Tuple, Union, cast

import gradio as gr
from PIL import Image

from src.adaptshot.config.settings import AdaptShotConfig
from src.adaptshot.core.learner import FewShotLearner


class AdaptShotUI:
    """Wraps FewShotLearner in a Gradio interface for pilots."""

    def __init__(self, config: AdaptShotConfig) -> None:
        self.config = config
        self.learner: Optional[FewShotLearner] = None
        self.support_dir: str = tempfile.mkdtemp(prefix="adaptshot_support_")

    def _ensure_learner(self) -> FewShotLearner:
        """Lazy-load learner if not initialized."""
        if self.learner is None:
            self.learner = FewShotLearner(config=self.config)
            # In a real pilot, we would load actual support images here.
            # For the UI demo, we keep it uninitialized until files are uploaded.
        return self.learner

    def load_support_files(self, files: List[object]) -> str:
        """Handle support image uploads and index embeddings."""
        if not files:
            return "❌ No files uploaded."
        
        paths: List[str] = []
        labels: List[str] = []
        for f in files:
            # Gradio returns NamedTuple or string path depending on version
            path = str(f.name if hasattr(f, "name") else f)
            # Simple heuristic: use folder name as label, or prompt user in advanced version
            label = os.path.basename(os.path.dirname(path)) or "unknown"
            paths.append(path)
            labels.append(label)

        try:
            learner = self._ensure_learner()
            learner.load_support_images(paths, labels)
            return f"✅ Indexed {len(paths)} support images for classes: {', '.join(set(labels))}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def predict_image(self, image: Union[str, Image.Image]) -> Tuple[str, float, float, str, str]:
        """Run prediction and return metadata for UI display."""
        if self.learner is None:
            return "Error", 0.0, 0.0, "None", "Learner not initialized. Upload support images first."

        try:
            result = self.learner.predict(image)
            
            # Save state for feedback routing
            self._last_result = result
            self._last_image_path = image if isinstance(image, str) else None

            return (
                str(result.prediction),
                float(result.raw_confidence),
                float(result.calibrated_confidence),
                result.act_action,
                "Ready for correction if needed."
            )
        except Exception as e:
            return "Error", 0.0, 0.0, "Error", str(e)

    def submit_correction(self, true_label: str, confidence_weight: float) -> str:
        """Route correction and return status."""
        if not hasattr(self, '_last_image_path') or self._last_image_path is None:
            return "❌ No prediction made yet to correct."
        
        try:
            assert self.learner is not None
            res = self.learner.correct(
                image_path=self._last_image_path,
                true_label=true_label,
                confidence_weight=confidence_weight
            )
            return f"✅ Correction routed! Fine-tuned: {res['fine_tuned']}, Buffer Size: {res['buffer_size']}"
        except Exception as e:
            return f"❌ Correction failed: {str(e)}"


def build_ui() -> gr.Blocks:
    """Construct the Gradio application."""
    config = AdaptShotConfig(device="cpu", seed=42)
    app = AdaptShotUI(config)

    with gr.Blocks(title="AdaptShot Pilot Dashboard") as ui:
        gr.Markdown("# 🌿 AdaptShot: Human-in-the-Loop Few-Shot Vision")
        gr.Markdown("Upload support images to build a class set, then predict on new images and provide feedback.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Load Support Set")
                support_files = gr.Files(file_count="multiple", label="Upload Images (Folders per Class)")
                load_btn = gr.Button("Index Support Set", variant="primary")
                load_status = gr.Textbox(label="Loading Status", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### 2. Inference")
                query_img = gr.Image(type="filepath", label="Upload Query Image")
                pred_btn = gr.Button("Predict", variant="primary")
                
                pred_output = gr.Label(label="Prediction")
                raw_conf = gr.Number(label="Raw Confidence", interactive=False)
                cal_conf = gr.Number(label="Calibrated Confidence", interactive=False)
                act_action = gr.Textbox(label="ACT Decision", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 3. Human Feedback")
                true_label_input = gr.Textbox(label="Correct Label (if wrong)")
                conf_weight = gr.Slider(0.0, 1.0, value=1.0, label="Your Confidence")
                correct_btn = gr.Button("Submit Correction")
                correction_status = gr.Textbox(label="Feedback Status", interactive=False)

        # Wiring
        load_btn.click(fn=app.load_support_files, inputs=support_files, outputs=load_status)
        pred_btn.click(
            fn=app.predict_image,
            inputs=query_img,
            outputs=[pred_output, raw_conf, cal_conf, act_action, correction_status]
        )
        correct_btn.click(
            fn=app.submit_correction,
            inputs=[true_label_input, conf_weight],
            outputs=correction_status
        )

    return cast(gr.Blocks, ui)


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", share=True)