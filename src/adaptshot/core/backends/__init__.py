"""Backend-agnostic feature extraction for AdaptShot.

Auto-detects the best available backend (ONNX Runtime → PyTorch) and provides
a unified ``extract_embedding()`` interface. The core library works with the
lightweight onnxruntime dependency by default; install ``adaptshot[torch]``
for training and fine-tuning support.

Priority order:
    1. ONNX Runtime (lightweight, fast install, CPU-optimized)
    2. PyTorch (full flexibility, required for training)
"""
