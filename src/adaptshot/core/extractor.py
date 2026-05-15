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
ImageInput = Union[str, np.ndarray, Image.Image, torch.Tensor]

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
    """Extract feature embedding from input image using a frozen backbone."""
    # Load backbone from registry
    if config.backbone not in BackboneRegistry:
        raise ValueError(f"Unknown backbone: {config.backbone}. Available: {list(BackboneRegistry.keys())}")

    backbone = BackboneRegistry[config.backbone]()
    backbone.fc = nn.Identity()
    backbone.to(config.device)
    backbone.eval()

    # Preprocess image
    preprocess = _get_preprocess_transform()
    
    # ✅ ADD THIS: Handle file paths
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
        
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] not in (1, 3):
            image = image.permute(2, 0, 1)
        image = transforms.ToPILImage()(image.cpu())

    # Apply transforms and add batch dimension
    image_tensor = preprocess(image).unsqueeze(0).to(config.device)

    with torch.no_grad():
        embedding = backbone(image_tensor).squeeze(0)

    return embedding.detach().cpu().numpy() if return_numpy else embedding