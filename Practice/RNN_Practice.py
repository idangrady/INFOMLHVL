# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:20:05 2021

@author: admin
"""

from Utilities_RNN import ALL_LETTERS,unicode_to_ascii, load_data,letter_to_tensor,line_to_tensor , random_training_example
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



class RNN(nn.Module):
    
    
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size
                 ):
        super(RNN, self).__init__()
        
        #initilize
        self.hidden_size = hidden_size
        self.i_h = nn.Linear(input_size + hidden_size,hidden_size)
        self.i_o = nn.Linear(input_size + hidden_size,output_size, bias = True)
        self.soft_max = nn.Softmax(dim = 1)
    
    def forward(self,X,h):
        combine = torch.cat((X,h), 1) # On layer 1
        hidden = (self.i_h(combine))
        output_y = self.soft_max(self.i_o(combine))
        return(output_y, hidden)
    
    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)
    

def category_to_category_output(output_tenser, categories):
    
    value_max, loc_ = output_tenser.topk(dim=-1, k=1)
    return(categories[loc_.item()])


def training(rnn, line_tens, category_tensor, loss_func, optimizer):
    """
    making one step
    """
    
    #one initilize hidden state
    hiddn = rnn.init_hidden()
    
    for letter in line_tens:
        #get each letter and send to the RNN
        output, hiddn = rnn(letter,hiddn)
    
    loss_ = loss_func(output, category_tensor)
    loss_.backward()
    optimizer.step()
    optimizer.zero_grad()
    

    return (output,loss_.item())
    

if __name__ == '__main__':
    
    print({f"all letters: {len(ALL_LETTERS)}"})
    
    category_lines, all_categories = load_data()


    n_hidden = 128
    rnn = RNN(len(ALL_LETTERS),n_hidden,  len(all_categories))

    X = random_training_example(category_lines, all_categories)
    
    #Sequence of words
 #   line = line_to_tensor('Albert')
    hidden_tens = rnn.init_hidden()
        
    
    #
    creterios = nn.NLLLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(params = rnn.parameters(),lr =learning_rate )
    
    
    
    currnt_loss = 0
    all_losses= []
    
    plot_step,ptrint_step = 1000,5000
    n_itters = 100000
    
    for _ in range(n_itters):
        
        category,line,category_tens,line_tens =random_training_example(category_lines, all_categories)
        
        
        output,loss = training(rnn,line_tens,category_tens, creterios,  optimizer )
        
        currnt_loss +=loss
        all_losses.append(loss)
        
        if _ %5000 ==0:
            guess = category_to_category_output(output,all_categories )
            condition = guess == category
            print(f"Guess: {guess},  correct : {category} ,  Name: {line}")
            print(loss)
            
        
        






















# =============================================================================
#     
#     input_tens= letter_to_tensor("B")
#     hidden_tens = rnn.init_hidden()
#     
#     
#     output, next_hidden = rnn(input_tens,hidden_tens )
# =============================================================================