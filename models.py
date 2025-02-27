#!/usr/bin/env python
# coding: utf-8

# # Defined the model

# In[ ]:


import torch
import torch.nn as nn

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


# In[ ]:


model = MLP(input_size=10, hidden_size=256, output_size=97, num_layers=8)
sample_input = torch.randn(1, 10)  
output = model(sample_input)
print(output)


# In[ ]:


print(output.shape)


# In[ ]:


import torch.nn.functional as F
print(F.softmax(output, dim=1))


# In[ ]:


sample_input = torch.randn(1, 10)
target = torch.tensor([1])  # Assume class 1 as target

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
output = model(sample_input)
loss = criterion(output, target)  # No need for unsqueeze()
loss.backward()
optimizer.step()

print("Loss after one step:", loss.item())


# In[ ]:


get_ipython().system('jupyter nbconvert --to script models.ipynb')
get_ipython().system('jupyter nbconvert --to script train.ipynb')
get_ipython().system('jupyter nbconvert --to script utils.ipynb')


# In[ ]:





# In[ ]:




