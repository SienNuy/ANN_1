import os
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt


#
# Contains the architecture of the network which will be trained
#
class CNN(torch.nn.Module):
    def __init__(self, x=None):
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


#
# Reads images from a directory, creates a data generator to load images batch by batch, and does some image
# pre-processing
#
class LoadData:
    def __init__(self, args):
        base_dir = './input_output'
        traindir = base_dir + '/data/15SceneData/train'
        valdir = base_dir + '/data/15SceneData/test'

        train_data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                                   transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(traindir, transform=train_data_transform)

        val_data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                                 transforms.ToTensor()])
        val_dataset = datasets.ImageFolder(valdir, transform=val_data_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)


#
# Trains the network and recordes the train and validation loss for each epoch
#
class TrainCNN:
    def __init__(self, model, data, device=None):
        # the network that should be trained
        self.model = model
        # Determines the loss function
        self.loss = self.loss_function()
        # Determines the optimazation algorithm
        self.optim = self.optim_function()
        # Dataset
        self.data = data
        # A list for recording the training loss
        self.train_loss_history = []
        # A list for recording the validation loss
        self.val_loss_history = []

        self.fit(device)

    def loss_function(self):
        return F.cross_entropy

    def optim_function(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def fit(self, device):
        epochs = 20
        for epoch in range(epochs):
            print(epoch)
            temp_loss = 0
            itr = 0
            for i, (xb, yb) in enumerate(self.data.train_loader):
                temp_loss += self.loss_batch(xb, yb, device)
                itr = i

            # The average training loss in each epoch is added to the list
            self.train_loss_history.append(temp_loss.item() / itr)
            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                itr = 0
                # This loop calculates the validity loss for each data set and the results are summed in the val_loss
                # variable
                for j, (xb, yb) in enumerate(self.data.val_loader):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    self.model = self.model.to(device)
                    valid_loss += self.loss(self.model(xb), yb)
                    itr = j
                # The average validation loss in each epoch is added to the list
                self.val_loss_history.append(valid_loss.item() / itr)

            print(epoch, valid_loss / len(self.data.val_loader))

    def loss_batch(self, xb, yb, device):
        self.model = self.model.to(device)
        xb = xb.to(device)
        yb = yb.to(device)
        # Feeding the network by an input batch xb, followed by calling the loss function
        loss = self.loss(self.model(xb), yb)
        if self.optim is not None:
            # Calculates the gradient of each weight respect with the loss
            loss.backward()
            # Updates the weights of the networks
            self.optim.step()
            # Sets the gradients to zero
            self.optim.zero_grad()
        return loss


#
# Plots the figures
#
def plot_trainval_loss(train, val):
    plt.plot(train, 'ro--', linewidth=2, markersize=12, label='Train Loss')
    plt.plot(val, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12, label='Validatin Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()


################################################################################3
from codes.argument_parse import my_arg
'''
creating a arument parser
'''
args = my_arg()

'''
load dataset
'''
data = LoadData(args)
'''
Creating the architecture of the network and design the training procedure.
'''
network = CNN()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = network.to(device)
train = TrainCNN(network, data, device)

print(train.train_loss_history)
print(train.val_loss_history)

plot_trainval_loss(train.train_loss_history, train.val_loss_history)
