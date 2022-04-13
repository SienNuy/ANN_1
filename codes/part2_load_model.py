from torchvision import models
import torch.nn as nn


class loadmodel():
    def __init__(self, classes):
        # Load a pre-trained network called vgg16
        self.model = models.vgg16(pretrained=True)
        print(self.model)

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, classes)
        print(self.model)

        count = 0
        for p in self.model.parameters():
            if count < 15:
                p.requires_grad=False
            count+=1

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
