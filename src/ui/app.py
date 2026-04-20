"""Gradio UI for AdaptShot real-time prediction and feedback updates."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.models.embedding import compute_cosine_similarity, extract_embedding
from src.training.feedback import ReplayBuffer, route_feedback
from src.training.incremental import incremental_fine_tune

__all__ = ["create_gradio_app", "launch_gradio_app"]


def _class_names(model: torch.nn.Module) -> List[str]:
    """Infer class names from classifier output width."""
    num_classes = int(model.fc.out_features)
    return [f"class_{idx}" for idx in range(num_classes)]


def _buffer_stats(buffer: ReplayBuffer) -> str:
    """Create compact buffer status string."""
    return f"Buffer size: {len(buffer)} / {buffer.capacity}"


def _prepare_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized input tensor for the backbone."""
    rgb_image = image.convert("RGB")
    resized = rgb_image.resize((128, 128))
    return to_tensor(resized).unsqueeze(0)


def create_gradio_app(
    model: torch.nn.Module,
    embedding_extractor: Callable[[torch.nn.Module, torch.Tensor], np.ndarray],
    replay_buffer: ReplayBuffer,
    device: torch.device,
) -> gr.Blocks:
    """
    Build Gradio Blocks app with prediction, confidence, NN view, and feedback.

    The app uses CPU-safe handlers and routes corrections into the replay buffer.
    """
    model.to(device)
    model.eval()
    classes = _class_names(model)

    def predict(
        image: Image.Image | None,
        state: Dict[str, Any],
    ) -> Tuple[Dict[str, float], float, Image.Image | None, str, str, Dict[str, Any]]:
        """Run model prediction and nearest-neighbor retrieval."""
        try:
            if image is None:
                return {}, 0.0, None, "Upload an image to run prediction.", _buffer_stats(replay_buffer), state

            input_tensor = _prepare_image(image).to(device=device, non_blocking=False)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            label_map = {classes[idx]: float(probs[idx]) for idx in range(len(classes))}

            embedding = embedding_extractor(model, input_tensor.squeeze(0).cpu())
            nearest_image: Image.Image | None = None
            status = f"Predicted {classes[pred_idx]} @ confidence={confidence:.3f}"

            support_embs, _, support_images = replay_buffer.get_batches()
            if support_embs:
                sim = compute_cosine_similarity(embedding, np.stack(support_embs))
                nn_idx = int(np.argmax(sim))
                if support_images[nn_idx] is not None:
                    nearest_image = support_images[nn_idx]
                status += f" | NN cosine={float(sim[nn_idx]):.3f}"

            new_state = {
                "embedding": embedding,
                "predicted_label": pred_idx,
                "image": image,
            }
            return label_map, confidence, nearest_image, status, _buffer_stats(replay_buffer), new_state
        except Exception as exc:  # pragma: no cover - defensive UI path
            return {}, 0.0, None, f"Prediction error: {exc}", _buffer_stats(replay_buffer), state

    def reinforce(
        state: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Handle correct feedback and trigger a small incremental update."""
        try:
            if not state or "embedding" not in state or "predicted_label" not in state:
                return "No prediction state found. Run prediction first.", _buffer_stats(replay_buffer)

            replay_buffer.add(
                embedding=np.asarray(state["embedding"], dtype=np.float32),
                label=int(state["predicted_label"]),
                image=state.get("image"),
            )
            loss = incremental_fine_tune(
                model=model,
                buffer=replay_buffer,
                fisher_dict=None,
                old_params=None,
                lam=0.1,
                lr=1e-4,
                epochs=1,
            )
            return f"Reinforced prediction; incremental loss={loss:.4f}", _buffer_stats(replay_buffer)
        except Exception as exc:  # pragma: no cover - defensive UI path
            return f"Feedback routing error: {exc}", _buffer_stats(replay_buffer)

    def correct_with_label(
        corrected_label: str,
        state: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Handle wrong feedback using selected corrected class."""
        try:
            if not state or "embedding" not in state or "predicted_label" not in state:
                return "No prediction state found. Run prediction first.", _buffer_stats(replay_buffer)

            if corrected_label not in classes:
                return "Select a valid corrected label.", _buffer_stats(replay_buffer)

            corrected_idx = classes.index(corrected_label)
            added = route_feedback(
                buffer=replay_buffer,
                embedding=np.asarray(state["embedding"], dtype=np.float32),
                predicted_label=int(state["predicted_label"]),
                corrected_label=corrected_idx,
                image=state.get("image"),
            )
            if not added:
                return "Correction matched predicted class; nothing added.", _buffer_stats(replay_buffer)

            loss = incremental_fine_tune(
                model=model,
                buffer=replay_buffer,
                fisher_dict=None,
                old_params=None,
                lam=0.1,
                lr=1e-4,
                epochs=2,
            )
            return f"Correction applied; incremental loss={loss:.4f}", _buffer_stats(replay_buffer)
        except Exception as exc:  # pragma: no cover - defensive UI path
            return f"Feedback routing error: {exc}", _buffer_stats(replay_buffer)

    with gr.Blocks(title="AdaptShot - Human-in-the-Loop Few-Shot") as app:
        gr.Markdown("## AdaptShot: Real-Time Few-Shot Prediction + Feedback")
        state = gr.State(value={})

        with gr.Row():
            image_input = gr.Image(
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                label="Input image",
            )
            nearest_neighbor = gr.Image(type="pil", label="Nearest replay neighbor")

        predict_button = gr.Button("Predict")
        prediction_label = gr.Label(label="Class probabilities")
        confidence_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.001,
            label="Confidence",
            interactive=False,
        )
        status_text = gr.Textbox(label="Status", value="Ready")
        buffer_stats = gr.Textbox(label="Replay Buffer", value=_buffer_stats(replay_buffer))

        with gr.Row():
            correct_button = gr.Button("✓ Correct / Reinforce")
            wrong_label = gr.Dropdown(choices=classes, value=classes[0], label="Correct class (if wrong)")
            wrong_button = gr.Button("✗ Wrong / Correct")

        predict_button.click(
            fn=predict,
            inputs=[image_input, state],
            outputs=[prediction_label, confidence_slider, nearest_neighbor, status_text, buffer_stats, state],
        )
        correct_button.click(
            fn=reinforce,
            inputs=[state],
            outputs=[status_text, buffer_stats],
        )
        wrong_button.click(
            fn=correct_with_label,
            inputs=[wrong_label, state],
            outputs=[status_text, buffer_stats],
        )

    return app


def launch_gradio_app(
    model: torch.nn.Module,
    replay_buffer: ReplayBuffer,
    device: torch.device = torch.device("cpu"),
) -> gr.Blocks:
    """Create and launch the AdaptShot Gradio app with required launch flags."""
    app = create_gradio_app(
        model=model,
        embedding_extractor=extract_embedding,
        replay_buffer=replay_buffer,
        device=device,
    )
    app.launch(
        share=True,
        inline=True,
        server_name="0.0.0.0",
        quiet=True,
        show_error=True,
    )
    return app
