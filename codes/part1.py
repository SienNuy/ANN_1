#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks
#
# Let's start by building a network that uses convolutional layers.

# In[1]:


import torch

# In[2]:


import torch.nn.functional as F

# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}!")


# In[4]:


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        # Connected layer
        self.fc = torch.nn.Linear(46656, 15)

    def forward(self, x):
        # A convolutional layer , followed by a relu layer
        x = F.relu(self.conv1(x))
        # A max pooling layer
        x = F.max_pool2d(x, 2)
        # A convolutional layer , followed by a relu layer
        x = F.relu(self.conv2(x))
        # A max pooling layer
        x = F.max_pool2d(x, 2)
        # Flats all the filters in a vector
        x = torch.flatten(x, 1)
        # A fully connected layer
        x = self.fc(x)
        # A softmax layer to map the output vector into the range [0 ,1] such that the summation of all elements will
        # be 1
        output = F.softmax(x, dim=1)
        return output


# Load the data.

# In[5]:


import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image


# In[6]:


class DataLoader:
    def __init__(self):
        base_dir = "input_output/data"
        train_dir = base_dir + "/15SceneData/train"
        val_dir = base_dir + "/15SceneData/test"

        train_data_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        train_dataset = datasets.ImageFolder(train_dir, transform=train_data_transform)

        val_data_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        val_dataset = datasets.ImageFolder(val_dir, transform=val_data_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)


# In[7]:


from torch import optim


# In[8]:


class Trainer:
    def __init__(self, model, data: DataLoader):
        self.model = model
        self.loss = F.cross_entropy
        self.optim = optim.SGD(self.model.parameters(), lr=0.2, momentum=0.9)
        self.data = data
        self.train_loss_history = []
        self.val_loss_history = []
        print("start fit")
        self.fit()

    def fit(self, epochs=20):
        for epoch in range(epochs):
            print(epoch)
            temp_loss = 0
            itr = 0
            for i, (xb, yb) in enumerate(self.data.train_loader, start=1):
                xb, yb = xb.to(device), yb.to(device)
                temp_loss += self.loss_batch(xb, yb)
                itr = i

            self.train_loss_history.append(temp_loss.item() / itr)
            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                itr = 0
                for j, (xb, yb) in enumerate(self.data.val_loader, start=1):
                    xb, yb = xb.to(device), yb.to(device)
                    valid_loss += self.loss(self.model(xb), yb)
                    itr = j
                self.val_loss_history.append(valid_loss.item() / itr)
            self.model.train()

    def loss_batch(self, xb, yb):
        loss = self.loss(self.model(xb), yb)
        if self.optim is not None:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return loss


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


def plot_train_val_loss(train, val):
    plt.plot(train, "ro--", linewidth=2, markersize=12, label="Train Loss")
    plt.plot(val, color="green", marker="o", linestyle="dashed", linewidth=2, markersize=12, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# In[11]:


data = DataLoader()
print("data loaded")
# In[ ]:


network = ConvNet()
print("nets made")
network = network.to(device)
print('idk done')
trainer = Trainer(network, data)
print("training done")
plot_train_val_loss(trainer.train_loss_history, trainer.val_loss_history)

# In[ ]:




