#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:59:24 2019

@author: tim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Puffin(nn.Module):
    def __init__(self, freq_axis, time_axis, no_classes):
        super(Puffin, self).__init__()
        
        self.freq_axis = freq_axis #input_dim
        self.time_axis = time_axis
        self.no_classes = no_classes
        
        self.input_features = 1664 #input_dim
        
        # Hyper parameters
        # Hidden dimensions and number of hidden layers 
        self.harmony = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Dropout(0.3),
        
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3,2), stride=(3,2)),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            )
        
        # batch_first=True shapes Tensors : batch_dim, seq_dim, feature_dim)
        self.lstm1 = nn.LSTM(self.input_features, 500, 3, dropout=0.5, batch_first=True)
       
        self.lstm2 = nn.LSTM(500, 200, 2, dropout=0.5, batch_first=True)
        
        
        
        self.fc = nn.Sequential(            
            nn.Linear(200, self.no_classes),
            nn.ReLU(),
            )

        
    def forward(self, x):
        out =  self.harmony(x)
        out = out.view(out.shape[0], -1, out.shape[3]).permute(0,2,1)   
        #print(out.shape)
        output_seq, hidden_state = self.lstm1(out)
        output_seq, hidden_state = self.lstm2(output_seq)

        last_output = output_seq[:, -1]
        out = self.fc(last_output)
        
        return out

def test():
    cnn = Puffin(256, 216, 100)
    img = torch.randn(2, 1, 256, 216)
    cnn(img)

if __name__=="__main__":
    from torchsummary import summary
    test()
