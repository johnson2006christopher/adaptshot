import torch, numpy as np
from src.training.trainer import evaluate
from src.evaluation.metrics import compute_ece
from src.models.network import create_model

def test_confidence_range(device):
    """Ensures confidence scores stay within [0, 1]."""
    model = create_model(num_classes=5, device=device)
    # Mock dataloader with dummy data
    dummy_ds = torch.utils.data.TensorDataset(
        torch.randn(10, 3, 128, 128), torch.randint(0, 5, (10,))
    )
    loader = torch.utils.data.DataLoader(dummy_ds, batch_size=5)
    acc, conf = evaluate(model, loader, device)
    assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds!"

def test_ece_perfect_calibration():
    """ECE should be ~0.0 when confidence perfectly matches accuracy."""
    # Mock: Many confident predictions that are perfectly accurate
    np.random.seed(42)
    n = 100
    confs = np.random.uniform(0.9, 1.0, n)  # High confidence
    accs = np.ones(n)  # Always correct
    ece = compute_ece(confs, accs, n_bins=10)
    # Allow small tolerance due to binning approximation
    assert ece < 0.1, f"ECE too high for perfect calibration: {ece}"

def test_ece_miscalibration():
    """ECE should increase when confidence diverges from accuracy."""
    confs = np.array([0.9, 0.9, 0.9, 0.9])
    accs = np.array([0.0, 0.0, 0.0, 0.0])  # Model is confident but wrong
    ece = compute_ece(confs, accs, n_bins=4)
    assert ece > 0.5, f"ECE too low for severe miscalibration: {ece}"
