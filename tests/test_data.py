import torch, numpy as np
from PIL import Image
from src.data.dataset import get_transforms, get_dataloaders

def test_transforms_shape():
    """Verifies transforms output correct tensor shape."""
    t = get_transforms(is_train=True)
    # Create dummy PIL Image (HWC format, uint8)
    dummy_np = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(dummy_np)
    out = t(dummy_pil)
    assert out.shape == (3, 128, 128)

def test_dataloader_shapes():
    """Verifies batch shapes match expected dimensions."""
    train_loader, test_loader = get_dataloaders(batch_size=16, shots=10)
    imgs, labels = next(iter(train_loader))
    assert imgs.shape == (16, 3, 128, 128)
    assert labels.shape == (16,)

def test_few_shot_constraint():
    """Ensures exact few-shot sampling: 50 train, 25 test."""
    train_loader, test_loader = get_dataloaders(shots=10)
    assert len(train_loader.dataset) == 50
    assert len(test_loader.dataset) == 25

def test_reproducibility():
    """Verifies identical seeds produce identical data splits."""
    loader1, _ = get_dataloaders(shots=5, seed=42)
    loader2, _ = get_dataloaders(shots=5, seed=42)
    idx1 = set(loader1.dataset.indices)
    idx2 = set(loader2.dataset.indices)
    assert idx1 == idx2, "Seed control failed: splits differ!"
