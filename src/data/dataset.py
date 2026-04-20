import torch
import torchvision
import torchvision.transforms as T
import random

def get_transforms(is_train=True):
    """Returns torchvision transforms for training or evaluation."""
    if is_train:
        return T.Compose([
            T.Resize((140, 140)),
            T.RandomCrop(128),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    else:
        return T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

def get_dataloaders(root="./data", batch_size=16, shots=10, seed=42):
    """
    Loads a few-shot subset of CIFAR10.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Define classes for this experiment
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer']
    
    # Load full dataset
    train_full = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=get_transforms(is_train=True))
    test_full  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=get_transforms(is_train=False))
    
    # Subsample logic
    def subset(dataset, n_per_class):
        indices = []
        for cls_idx in range(len(classes)):
            cls_indices = [i for i, t in enumerate(dataset.targets) if t == cls_idx]
            random.shuffle(cls_indices)
            indices.extend(cls_indices[:n_per_class])
        return torch.utils.data.Subset(dataset, indices)
    
    train_set = subset(train_full, n_per_class=shots)
    test_set  = subset(test_full, n_per_class=5) # Fixed 5 for testing
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"Data loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
