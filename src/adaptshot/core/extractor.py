"""Frozen backbone feature extraction with TorchScript compatibility."""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from ..config.settings import AdaptShotConfig

# Type alias for flexible image input
ImageInput = Union[np.ndarray, Image.Image, torch.Tensor]

# Registry for backbone factories (extensible without modifying core logic)
BackboneRegistry = {
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": lambda: models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    ),
}


def _get_preprocess_transform(img_size: int = 224) -> transforms.Compose:
    """Return standard preprocessing transforms for ImageNet-pretrained backbones."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_embedding(
    image: ImageInput,
    config: AdaptShotConfig,
    return_numpy: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Extract feature embedding from input image using a frozen backbone.

    Args:
        image: Input as numpy array (HWC), PIL Image, or torch tensor (CHW or HWC)
        config: AdaptShotConfig with backbone, device, and pipeline settings
        return_numpy: If True, return numpy array; else return torch.Tensor

    Returns:
        Embedding tensor/array of shape (embedding_dim,). ResNet18 -> 512, MobileNetV3Small -> 576.
    """
    # Load backbone from registry
    if config.backbone not in BackboneRegistry:
        raise ValueError(f"Unknown backbone: {config.backbone}. Available: {list(BackboneRegistry.keys())}")

    backbone = BackboneRegistry[config.backbone]()

    # Remove classification head to extract features from global avgpool
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        # MobileNet variant
        backbone.classifier = nn.Identity()

    backbone.to(config.device)
    backbone.eval()

    # Preprocess image to match ImageNet training distribution
    preprocess = _get_preprocess_transform()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        # Ensure CHW format before conversion
        if image.dim() == 3 and image.shape[0] not in (1, 3):
            image = image.permute(2, 0, 1)
        image = transforms.ToPILImage()(image.cpu())

    # Apply transforms and add batch dimension [C, H, W] -> [1, C, H, W]
    image_tensor = preprocess(image).unsqueeze(0).to(config.device)

    # Extract embedding (no gradients computed)
    with torch.no_grad():
        embedding = backbone(image_tensor)  # Shape: [1, embedding_dim]
        embedding = embedding.squeeze(0)    # Shape: [embedding_dim]

    if return_numpy:
        return embedding.detach().cpu().numpy()
    return embedding