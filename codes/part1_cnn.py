#
# Contains the architecture of the network which will be trained
#
from torch import nn, flatten
import torch.nn.functional as F


class cnn(nn.Module):
    def __init__(self, x=None):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        # Connected layer
        self.fc = nn.Linear(46656, 15)
        self.dropout = nn.Dropout(0.01)  # it defines a dropout layer which dropouts 25 percent of nodes

    def forward(self, x):
        x = F.relu(self.conv1(x))  # A convolutional layer , followed by a relu layer
        x = F.max_pool2d(x, 2, stride=2)  # A max pooling layer
        x = F.relu(self.conv2(x))  # A convolutional layer , followed by a relu layer
        x = self.dropout(x)  # dropout 25 percent of nodes in tensor x
        x = F.max_pool2d(x, 2)  # A max pooling layer
        x = flatten(x, 1)  # Flats all the filters in a vector
        x = self.fc(x)  # A fully connected layer
        output = F.softmax(x, dim=1)  # A softmax layer to map the output vector into the range [0 ,1] such that the
        # summation of all elements will be 1
        return output
