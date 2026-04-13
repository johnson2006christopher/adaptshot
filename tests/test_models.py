import torch
from src.models.network import create_model

def test_model_creation(device):
    """Verifies model loads and moves to correct device."""
    model = create_model(num_classes=5, device=device)
    assert next(model.parameters()).device.type == device.type

def test_backbone_frozen(device):
    """Ensures ResNet18 backbone is frozen (requires_grad=False)."""
    model = create_model(num_classes=5, device=device)
    for name, param in model.named_parameters():
        if "fc" not in name:  # All except classifier head
            assert not param.requires_grad, f"Backbone param {name} is not frozen!"

def test_head_trainable(device):
    """Ensures only the classifier head is trainable."""
    model = create_model(num_classes=5, device=device)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 2  # fc.weight and fc.bias
    assert all("fc" in n for n in trainable)

def test_output_shape(device, dummy_batch):
    """Verifies forward pass outputs correct shape: [batch, num_classes]."""
    model = create_model(num_classes=5, device=device)
    with torch.no_grad():
        out = model(dummy_batch.to(device))
    assert out.shape == (4, 5)
