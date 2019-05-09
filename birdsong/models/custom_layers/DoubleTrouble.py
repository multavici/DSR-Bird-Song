#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:08:59 2019

@author: tim
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleTrouble(nn.Module):
    def __init__(self):
        super(DoubleTrouble, self).__init__()
        
        # batch_first=True shapes Tensors : batch_dim, seq_dim, feature_dim)
        
        
        self.summary = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        
        self.harmony = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=(125, 1), stride=1),            
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #nn.Dropout(0.3),
            )
        
        self.lstm1 = nn.RNN(512, 256, 3, dropout=0.5, batch_first=True)
        
        self.time_summary = nn.Sequential(
            nn.Conv1d(105, 64, kernel_size=252))
        
        self.fc = nn.Sequential(            
            nn.Linear(320, 100),
            #nn.ReLU(),
            )
        
    def forward(self, x):
        x = self.summary(x)
        x = self.harmony(x)
        x = x.view(x.shape[0], -1, x.shape[3]).permute(0,2,1)   
        
        output_seq, hidden_state = self.lstm1(x)
        print(output_seq.shape)#output_seq = output_seq.permute(0,2,1)   
        
        o = self.time_summary(output_seq)
        
        print(o.shape)
        o = o.view(o.shape[0], -1)
        print(o.shape)
        
        #output_seq, hidden_state = self.lstm2(output_seq)

        
        out = self.fc(o)
        print(out)
        return out

def test():
    cnn = DoubleTrouble()
    img = torch.randn(2, 1, 256, 216)
    cnn(img)

if __name__=="__main__":

    test()

