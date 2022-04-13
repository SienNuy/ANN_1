import torch.nn.functional as F
from torch import optim
import torch


class train_cnn():
    def __init__(self, model, data, device=None):
        self.model = model  # the network that should be trained
        self.loss =  self.loss_function()  # Determines the loss function
        self.optim = self.optim_function()  # Determines the optimazation algorithm
        self.data = data  # Dataset
        self.train_loss_history = []
        self.val_loss_history = []

        print("start fit")
        self.fit(device)

################################################################################3
    def loss_function(self):
        return  F.cross_entropy

    def optim_function(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def fit(self, device):
        epochs = 20
        for epoch in range(epochs):
            temp_loss = 0
            itr = 0
            jnvs= self.data.train_loader
            for i, (xb, yb) in enumerate(self.data.train_loader):
                temp_loss += self.loss_batch(xb, yb,device)
                itr = i
                print(itr)

            self.train_loss_history.append(temp_loss.item()/itr)
            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                itr = 0
                for j, (xb, yb) in enumerate(self.data.val_loader):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    self.model = self.model.to(device)
                    valid_loss += self.loss(self.model(xb), yb)
                    itr = j

                self.val_loss_history.append(valid_loss.item() / itr)


            print(epoch, valid_loss / len(self.data.val_loader))

    def loss_batch(self, xb, yb,device):
        self.model = self.model.to(device)
        xb = xb.to(device)
        yb = yb.to(device)
        loss = self.loss(self.model(xb), yb)
        if self.optim is not None:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        return loss