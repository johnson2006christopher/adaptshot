import sys

with open('src/adaptshot/training/finetune.py', 'r') as f:
    content = f.read()

old = """class CAEWCFinetuner:
    \"\"\"
    Correction-Aware Elastic Weight Consolidation (CA-EWC) fine-tuner.

    Fine-tunes the classification head using new corrections while penalizing 
    changes to weights deemed important by past Fisher Information.
    The penalty strength is modulated by the confidence weight of the human corrections:
    - High confidence (1.0) → Reduced penalty (Model adapts freely to the strong signal)
    - Low confidence (0.0) → Full penalty (Model stays conservative to preserve existing knowledge)
    \"\"\""""

new = """class CAEWCFinetuner:
    \"\"\"
    Correction-Aware Head-Only Fine-Tuning via Fisher Information regularization.

    IMPORTANT SCOPE NOTE (v0.2.0): This fine-tuner operates ONLY on the
    classification head — a single nn.Linear(embedding_dim, n_classes) layer
    containing ~(embedding_dim * n_classes) parameters (e.g., 2560 for 5-way
    with ResNet-18's 512-dim embeddings). It does NOT fine-tune the frozen
    backbone (ResNet/MobileNet). The term \"Elastic Weight Consolidation\"
    here refers to the Fisher-weighted regularization applied to these ~2K
    head parameters, not a full-network EWC implementation.

    For full backbone fine-tuning, use a dedicated training pipeline with
    GPU acceleration; this head-only approach is intentionally lightweight
    for CPU-first, resource-constrained environments.

    The penalty strength is modulated by the confidence weight of human corrections:
    - High confidence (1.0) -> Reduced penalty (Model adapts freely to the strong signal)
    - Low confidence (0.0) -> Full penalty (Model stays conservative to preserve existing knowledge)
    \"\"\""""

if old in content:
    content = content.replace(old, new)
    with open('src/adaptshot/training/finetune.py', 'w') as f:
        f.write(content)
    print('CA-EWC docstring: Replaced successfully')
else:
    print('CA-EWC docstring: Not found')
