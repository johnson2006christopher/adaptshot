# AdaptShot

**USP:** A zero-config, human-in-the-loop few-shot system that learns from corrections while guaranteeing calibrated uncertainty, deployable on CPU with fewer than 50 images per class.

AdaptShot is a research-grade Python package for few-shot classification, calibration tracking, active feedback routing, and continual adaptation with EWC regularization.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
python -m src.evaluation.benchmark --dry-run
```

To launch the UI:

```bash
python - <<'EOF'
import torch
from src.models.network import create_fewshot_model
from src.training.feedback import ReplayBuffer
from src.ui.app import launch_gradio_app

model = create_fewshot_model(num_classes=5, device=torch.device("cpu"))
buffer = ReplayBuffer(capacity=100)
launch_gradio_app(model=model, replay_buffer=buffer, device=torch.device("cpu"))
EOF
```

## Architecture

```text
Input Image
   |
   v
ResNet18 Backbone (frozen) ----> 512-d Embedding ----> Cosine Retrieval
   |
   v
Linear Head (trainable) ----> Probabilities + Confidence
   |
   v
Human Feedback (✓ / ✗) ----> Replay Buffer (FIFO) ----> Incremental Fine-Tune + EWC
```

## Phase 1-5 Features and Success Metrics

- **Phase 1:** deterministic runtime, package scaffold, CPU-safe tests
- **Phase 2:** deterministic N-way K-shot sampler, conservative augmentations, deterministic loader
- **Phase 3:** frozen ResNet18 + trainable head, 512-d embeddings, cosine similarity, ECE metric
- **Phase 4:** replay buffer, feedback routing, incremental fine-tuning, EWC penalty
- **Phase 5:** Gradio UI with live feedback loop, benchmark CLI, arXiv/repro docs

Target metrics:
- Few-shot accuracy: `>75%` on 5-shot CIFAR subset
- Calibration: `ECE < 0.05`
- Inference latency: `<50 ms` per image on CPU
- Feedback efficiency: each correction should improve subsequent behavior

## Reproduce Commands

```bash
# deterministic setup check
python -c "from src import configure_runtime; configure_runtime(seed=42, deterministic=True); print('runtime ready')"

# full tests
pytest tests/

# benchmarks
python -m src.evaluation.benchmark --dry-run

# import checks
python -c "from src.ui.app import create_gradio_app; print('ui import ok')"
python -c "from src.evaluation.benchmark import run_full_benchmark; print('benchmark import ok')"
```

Detailed step-by-step reproducibility instructions: `docs/REPRODUCIBILITY.md`.

## Citation

```bibtex
@misc{adaptshot2026,
  title={AdaptShot: Zero-Config Human-in-the-Loop Few-Shot Learning with Calibrated Uncertainty},
  author={AdaptShot Contributors},
  year={2026},
  howpublished={\url{https://github.com/johnson2006christopher/adaptshot}},
  note={Open-source research artifact}
}
```

## License

MIT License. See `LICENSE`.

## Acknowledgment

AdaptShot is built as an open research artifact focused on AI safety, few-shot learning, and human-AI alignment.

## Contributing

Contributions are welcome through issues and pull requests. Please include:
- deterministic reproduction steps,
- CPU-safe test updates,
- and clear metric impact notes (accuracy, ECE, latency).
