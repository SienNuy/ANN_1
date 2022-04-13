from codes.part1_cnn import cnn
from codes.part1_cnn_train import train_cnn
from codes.argument_parse import my_arg
from codes.part1_setup_data import load_data
from codes.part1_statistics import plot_trainval_loss
import torch
'''
creating a arument parser
'''
args = my_arg()

'''
load dataset
'''
data = load_data(args)
'''
Creating the architecture of the network and design the training procedure.
'''
network = cnn()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = network.to(device)
print("start training")
train = train_cnn(network, data, device)

print(train.train_loss_history)
print(train.val_loss_history)

plot_trainval_loss(train.train_loss_history, train.val_loss_history)
