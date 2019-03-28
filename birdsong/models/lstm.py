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
        
        self.input_features = 64 #input_dim
        self.seq_length = 52 
        
        # Hyper parameters
        # Hidden dimensions and number of hidden layers
        self.hidden_dim = 200 #500
        self.layer_dim = 3 #7
        
        self.harmony = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5,3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3,2), stride=(3,2)),
            
            nn.Conv2d(16, 32, kernel_size=(5,3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3,2), stride=(3,2)),
            
            nn.Conv2d(32, 64, kernel_size=(3,1), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            
            nn.Conv2d(64, 64, kernel_size=(3,1), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            
            nn.Conv2d(64, 64, kernel_size=(2,1), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        
        # batch_first=True shapes Tensors : batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(self.input_features, self.hidden_dim, self.layer_dim, dropout=0.5, batch_first=True)
        
        
        self.fc = nn.Linear(self.hidden_dim, self.no_classes)
        
    def forward(self, x):
        out =  self.harmony(x).squeeze(2).permute(0,2,1)   
        #Out shape: batch_dim, 52 time_steps, 64 timbral_features

        output_seq, hidden_state = self.lstm(out)
        last_output = output_seq[:, -1]
        out = self.fc(last_output)
        
        return out

def test():
    image = torch.randn(64, 1, 256, 216)
    lstm = LstmModel(256, 216, 100)
    output = lstm(image)
    
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    test()
