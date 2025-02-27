#!/usr/bin/env python
# coding: utf-8

# # Implemented training

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Model Definition 
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Compute Layer-wise Feature Rank 
def compute_layerwise_feature_rank(model, data_loader, device, tol=1e-5):
    """
    Compute the rank of activations for each Linear layer in the model using PyTorch.
    """
    model.eval()
    layer_ranks = {}
    with torch.no_grad():
        for i, layer in enumerate(model.network):
            if isinstance(layer, nn.Linear):
                layer_ranks[f'Layer_{i}'] = []

        for data, _ in data_loader:
            data = data.to(device)
            x = data
            for i, layer in enumerate(model.network):
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    # ✅ Compute rank using PyTorch
                    rank = torch.linalg.matrix_rank(x, tol=tol).item()
                    layer_ranks[f'Layer_{i}'].append(rank)

    # ✅ Compute average rank per layer
    avg_ranks = {layer: np.mean(ranks) for layer, ranks in layer_ranks.items()}
    return avg_ranks


# Dynamic Gradient Filtering 
def dynamic_filtering(gradients, epoch, total_epochs, validation_loss=None):
    base_threshold = 0.01
    if validation_loss and validation_loss < 0.5:
        base_threshold /= 2
    threshold = base_threshold * (0.0001 / base_threshold) ** (epoch / total_epochs)
    return [g * (g.abs() >= threshold) for g in gradients]

# Training Loop
def train(model, data_loader, criterion, optimizer, epochs, apply_filtering=False, device='cpu'):
    model.to(device)
    accuracy_list, rank_list_per_layer = [], []

    for epoch in range(epochs):
        model.train()
        total_correct, total_samples = 0, 0
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            if apply_filtering:
                with torch.no_grad():
                    filtered_grads = dynamic_filtering([p.grad for p in model.parameters()], epoch, epochs)
                    for param, filtered_grad in zip(model.parameters(), filtered_grads):
                        param.grad = filtered_grad

            optimizer.step()
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

        accuracy = total_correct / total_samples
        accuracy_list.append(accuracy)

        if epoch % 10 == 0:
            layer_ranks = compute_layerwise_feature_rank(model, data_loader, device)
            rank_list_per_layer.append(layer_ranks)
            # ✅ Inspect the rank data structure
            print(f"Ranks sample at Epoch {epoch}: {rank_list_per_layer[-1]}")


    print(f"Returning from train: accuracy_list length={len(accuracy_list)}, rank_list_per_layer length={len(rank_list_per_layer)}")
    return accuracy_list, rank_list_per_layer


# In[ ]:




