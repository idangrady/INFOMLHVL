# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 11:07:06 2021

@author: admin
"""
import numpy as np
import torch as tt
import torch.nn as nn

from sklearn import datasets
import matplotlib.pyplot as plt


# =============================================================================
# 
# 
# class numpy_imp:
#     """
#     implementation with numpy
#     simple function: f(x) = x**2 + 5
#     """
#     
#     #foreard
#     def forward(w,x):
#         return np.dot(x,w)
#     
#     #gradient
#     def gradient_num(x,y,y_predict):
#         """
#         compute the gradient manually
#         """
#         return(np.dot(2*x, (y-y_predict))).mean()
#     
#     #loss
#     def loss_num(x,w,y):
#         return((np.dot(w,x) -y)**2).mean()
# =============================================================================
    
def forward(w,x):
    return w*x

def loss_num(x,w,y):
    return((np.dot(w,x) -y)**2).mean()

def gradient_num(x,y,y_predict):
    """
    compute the gradient manually
    """
    return(np.dot(2*x, (y-y_predict))).mean()




def learn_(w,x,y,num_itter =40, lr =0.1):
    
    for epoch in range(num_itter):
        if epoch %5 ==0:
            predict = forward(w,x)
            loss_ = loss_num(x,w,y)
            g_w = gradient_num(x,y,predict)
            #update vakues: 
            w = w - lr*(g_w)


            print(f"w: {w}")
            print(f"loss: {loss_}")
            
        



def forward(w,x):
    return w*x


def loss_num(y_pred,y):
    return((y_pred-y)**2).mean()


def part_2():
    X =tt.arange((4),dtype = tt.float32)
    Y =X*2 
    
    w = tt.tensor(0.0,dtype = tt.float32,requires_grad=True)
    
    
    n_itters =30
    lr = 0.1
    for epoch in range(n_itters):
        predict = forward(w,X)
        
        loss = loss_num(predict,Y)
        
        
        loss.backward()  #dl/dw
        
        with tt.no_grad():
            w-= lr * w.grad
        
        if epoch %5 ==0:
            print(f"w: {w}")
            print(f"loss: {loss}")
            
        w.grad.zero_ ()
        


def part_3():
    n_itters =30
    lr = 0.1
    X =tt.arange((4),dtype = tt.float32)
    Y =X*2 
    
    w = tt.tensor(0.0,dtype = tt.float32,requires_grad=True)
    predict = forward(w,X)
    loss = nn.MSELoss()
    optimizer =tt.optim.SGD(params = [w], lr = lr)
    
    for epoch in range(n_itters):
        
        
        predict = forward(w,X)
        
        loss_ = loss(Y,predict)
        
        
        loss_.backward()
        if epoch %5 ==0:
            print(f"w: {w}")
            print(f"loss: {loss}")
            
        optimizer.step()
        optimizer.zero_grad()
        
    
def part_4():
    
    

    n_itters =1000
    lr = 0.1
    X =tt.arange((4),dtype = tt.float32).unsqueeze(0).T
    Y = X*2
    x = tt.tensor([[1],[2],[3],[4]])
    y = tt.tensor([[2],[4],[6],[8]])
    
    num_rows, num_feat = X.shape
    input_,outpit_  = num_feat,num_feat
    
    model = nn.Linear(input_,outpit_ )
    
    predict = model(X)
     
    loss = nn.MSELoss()

    optimizer =tt.optim.SGD(params = model.parameters(), lr = lr)
    
    for epoch in range(n_itters):

        loss_ = loss(Y,predict)
        
        loss_.backward()
        if epoch %100 ==0:
            w,b = model.parameters()
            print(f"w: {w}")
            print(f"loss: {loss}")
        
        optimizer.step()
        
        optimizer.zero_grad()




def linearRegression(n_itters=100):
    x_numpy,y_numpy = datasets.make_regression(n_samples =100, n_features = 1, noise = 20 , random_state = 1)
    
    print(y_numpy.shape)
    
    x_tt = tt.from_numpy(x_numpy.astype(np.float32))
    Y_tt = tt.from_numpy(y_numpy.astype(np.float32)).unsqueeze(0)
    print(Y_tt)
    
        #model

    n_rows,n_feat = x_tt.shape
    model = nn.Linear(n_feat,1)
    
    
    #loss and optimizer
    loss = nn.MSELoss()
    optimizer = tt.optim.SGD(params=model.parameters(),lr =0.1)
    
    
    for epoch in range(n_itters):
        
        predict = model(x_tt)

        loss_ = loss(predict, Y_tt)
        
        loss_.backward()
        
        
        if epoch %100 ==0:
            w,b = model.parameters()
            print(f"w: {w}")
            print(f"loss: {loss}")
            
        optimizer.step()
        
        optimizer.zero_grad()

    
    
    
    
    
    #training loop

if __name__ == "__main__":
    linearRegression()
    