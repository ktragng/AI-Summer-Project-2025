#!/usr/bin/env python
# coding: utf-8

# # Prepared the dataset

# In[1]:


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def generate_modular_data(num_samples=5000, mod_value=97):
    X = np.random.randint(0, mod_value, size=(num_samples, 2))  # Two input numbers
    y = (X[:, 0] * X[:, 1]) % mod_value  # Modular multiplication task
    
    # Normalize inputs to mean 0, variance 1
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Function to create DataLoader
def get_dataloader(batch_size=64):
    X_train, y_train = generate_modular_data(num_samples=5000)
    dataset = TensorDataset(X_train, y_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[ ]:




