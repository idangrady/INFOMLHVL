# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:46:17 2021

@author: admin
"""

import torch 
import torchvision
from torch.utils.data import Dataset,DataLoader

import numpy as np
import math
import pandas as pd



class WineData(Dataset):
    
    
    def __init__(self,loc, transform = None):
        #load data   
        x_y = np.array(pd.read_csv(loc))
        self.x= (x_y[:,1:])
        print(self.x.shape)
        self.y = (x_y[:,0])   
        self.transform =transform
        self.num_sam = x_y.shape[0]

        
        #split_data
        
    
    def __getitem__(self,idx):
        sample=  self.x[idx], np.array(self.y[idx])

        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.num_sam
    
class to_Tensor:
    def __call__(self, sample):
        x_, y_ = sample
        return (torch.from_numpy(x_), torch.from_numpy(y_))
    

class multi:
    def __init__(self, factor):
        self.factor = factor
    
    
    
    def __call__(self, sample):
        x_i, y_i = sample   
        return(x_i ** self.factor , y_i )
    
    
    def transform(self):
        return torch.from_numpy()
    
    
def first_tutu():
    location ="https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv"

    data_s =WineData(location)
    
  
    
    # training loop
    num_epoch = 2
    total_num_sam = len(data_s)
    batch_size_ = 4
    num_itter = total_num_sam // batch_size_
    
    
    
    dataLoader = DataLoader(dataset=data_s, batch_size=batch_size_, shuffle=True)
    dataitter = iter(dataLoader)
    data = dataitter.next()
    features,labels = data
    print(features, labels)
    print(dataLoader)
    
    
    for epoch in range(num_epoch):
        
        #loop over the train loader
        
        for i, (inputs,label) in enumerate(dataLoader):
            if i % 10 ==0:
                print(f"{epoch}\ {inputs.shape} \ {label}")    

if __name__ == '__main__':
    location ="https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv"

    data_set = WineData(location, transform = to_Tensor())
    first_data = data_set[0]
    features, label = first_data
    print((features),type(label))
    


    compose_func = torchvision.transforms.Compose([to_Tensor(), multi(2)])
    data_set = WineData(location, transform = compose_func)
    first_data = data_set[0]
    features, label = first_data
    print((features),type(label))
    

