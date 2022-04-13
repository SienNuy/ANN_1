import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

class load_data():
    def __init__(self, args):
        base_dir = './input_output'
        traindir = base_dir + '/data/15SceneData/train'
        valdir = base_dir + '/data/15SceneData/test'
        train_data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])
                                                   ])
        train_dataset = datasets.ImageFolder(traindir, transform=train_data_transform)
        val_data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
        val_dataset = datasets.ImageFolder(valdir, transform=val_data_transform)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
