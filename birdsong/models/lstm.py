#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:59:17 2019

@author: ssharma
"""

import torch
import torch.nn as nn
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LstmModel(nn.Module):

    def __init__(self, freq_axis, time_axis, no_classes):
        super(LstmModel, self).__init__()
        
        self.freq_axis = freq_axis #input_dim
        self.time_axis = time_axis
        self.no_classes = no_classes
        
        self.input_dim = 42 #input_dim
        self.seq_length = 36
        
        # Hyper parameters
        # Hidden dimensions and number of hidden layers
        self.hidden_dim = 200 #500
        self.layer_dim = 5 #7


        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(6,6), stride =(6,6))
           )

             
        # batch_first=True shapes Tensors : batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=0.5, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, self.no_classes)
        
        

    def forward(self, x):
        #reshape 
        out =  self.layer1(x)         
        out = out.view(-1, self.seq_length, self.input_dim).type(torch.float)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim).requires_grad_().to(device)

        # 25 time steps
        out, (hn, cn) = self.lstm(out, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        
        return out

def test():
    image = torch.randn(1, 1, 256, 216)
    lstm = LstmModel(256, 216, 10)
    output = lstm(image)
    
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    test()
