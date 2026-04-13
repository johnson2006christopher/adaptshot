import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluates the model and computes metrics."""
    model.eval()
    correct = 0
    total = 0
    confidences = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Compute probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Compute Confidence (1 - Normalized Entropy)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            max_entropy = torch.log(torch.tensor(float(probs.size(1))))
            conf = 1.0 - (entropy / max_entropy)
            confidences.extend(conf.cpu().numpy())
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    avg_confidence = np.mean(confidences)
    return accuracy, avg_confidence
