import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

def create_model(num_classes=5, device=None):
    """
    Creates a ResNet18 model with a pre-trained backbone and a new classifier head.
    
    Args:
        num_classes (int): Number of output classes.
        device (torch.device): Target device (cuda/cpu).
    
    Returns:
        torch.nn.Module: The configured model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Replace the final fully connected layer
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Freeze backbone parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier head
    for param in model.fc.parameters():
        param.requires_grad = True
        
    model.to(device)
    return model

if __name__ == "__main__":
    model = create_model()
    print(f"Model created on {next(model.parameters()).device}")
