import numpy as np
import torch 
import torch.nn as nn

from sklearn import datasets
import matplotlib.pyplot as plt
import torchvision 
#import torchvision.transforms as transforms
from Data_loader import  multi


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# implement __call__(self, sample)
class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

#hyperparameters

input_size = (28* 28) # should be flattern after
hidden_size = 1000
num_classes=10
num_epoche =1
batch_size = 100
lr = 0.001

#uplaod

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


examples = (iter(train_loader))
print(next(examples))
# =============================================================================
# examples = iter(test_loader)
# example_data, example_targets = examples.next()
# =============================================================================




class NeuralNet(nn.Module):
    
    def __init__(self, num_input, hidden_size, output_size):
        super(NeuralNet, self).__init__()
# =============================================================================
#         self.num_input = num_input
#         self.num_outputs = num_output
# =============================================================================
        self.l1 = nn.Linear(num_input,hidden_size)
        self.relu = nn.ReLU()
        self.l_2 = nn.Linear(hidden_size, output_size)
        self.sigmoid =nn 