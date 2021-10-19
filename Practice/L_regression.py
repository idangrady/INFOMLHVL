
import torch as tt
import torch.nn as nn
import numpy as np
from  sklearn import datasets
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

np.set_printoptions(suppress=True)

def importfile():
    return(pd.read_csv("https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv",sep = ',' ))
    

#prepare
def preprosess():
    df = datasets.load_breast_cancer()
    X, y = df.data , df.target
    X_train, X_test,Y_train,Y_test = train_test_split(X, y, test_size= 0.8, random_state=6)
    
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    [X_train, X_test,Y_train,Y_test]  =tt.from_numpy(X_train.astype(np.float32)),tt.from_numpy(X_test.astype(np.float32)),tt.from_numpy(Y_train.astype(np.float32)),tt.from_numpy(Y_test.astype(np.float32))

    print(f"before: {Y_train.shape}")
    Y_train,Y_test = Y_train.unsqueeze(0).T, Y_test.unsqueeze(0).T
    print(f"After: {Y_train.shape}")
    #transform the data zero mean
    

    return[X_train, X_test,Y_train,Y_test]



class logistic_regression(nn.Module):
    
    def __init__(self, num_input_feature ):
        super(logistic_regression, self).__init__()
            
        #num_rows,num_feat = X.shape
        self.model  =  nn.Linear(num_input_feature, 1)
        
    def forward(self,X):
        predict = tt.sigmoid(self.model(X))
        return(predict)

class WineData(Dataset):
    
    def __init__(self):
        xy  = importfile()
        xy =xy.to_numpy()
        self.n_samples = xy.shape[0]
        
        
        self.x_data = tt.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = tt.from_numpy(xy[:, [0]]) # size [n_samples, 1]
        
    
    def __getitem__(self, index):
        return(self.data[index],self.y_data[index] )        
        
    
       # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    
def linear_regress(n_iters):
    
    #load
    [X_train, X_test,Y_train,Y_test] =  preprosess()
    
    
    num_rows,num_feat = X_train.shape
    
    # model
    imput_data, output_data = num_feat,num_feat
    
    model = logistic_regression(imput_data)
    
    # loss and optimizer
    criterion = nn.BCELoss()
    optimizer = tt.optim.SGD(model.parameters(), lr =0.01 )
    
        
    # loop
    for epoch in range(n_iters):
        
        predict = model(X_train)
        
        loss =criterion(predict,Y_train)
        
        if epoch %10 ==0:
            print(f"{epoch, loss.item(),}")
        
        loss.backward()
        
        
        optimizer.step()
        optimizer.zero_grad()


def transform_():
    
    data = WineData()

    
    train_da = DataLoader(dataset = data,batch_size = 20, shuffle = True, num_workers =2 )
    print(train_da)
    dataiter = iter(train_da)
    data = dataiter.next()
    features, labels = data
    print(features, labels)
    
    # Dummy Training loop
    num_epochs = 2
    total_samples = len(data)
    n_iterations = math.ceil(total_samples/4)
    print(total_samples, n_iterations)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_da):
            
            # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
            # Run your training process
            if (i+1) % 5 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')


def main():
    transform_()



main()