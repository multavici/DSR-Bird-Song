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
    def __init__(self, freq_axis=256, time_axis=216,  no_classes=10):
        super(Puffin, self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.__name__='Puffin'

        # Frequency block
        self.frequency = nn.Sequential(

            # One filter, two frequency bands over time
            nn.Conv1d(freq_axis, freq_axis//2, kernel_size=5, stride=1),
            nn.BatchNorm1d(freq_axis//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=3, stride=3),
            
            # One filter, two frequency bands over time
            nn.Conv1d(freq_axis//2, freq_axis//4, kernel_size=5, stride=1),
            nn.BatchNorm1d(freq_axis//4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=3, stride=3),
            
            nn.Conv1d(freq_axis//4, freq_axis//8, kernel_size=22, stride=1),
            nn.BatchNorm1d(freq_axis//8),
            nn.ReLU(),
            nn.Dropout(0.3),
            )
        
        # Time block
        self.time = nn.Sequential(

            nn.Conv1d(time_axis, time_axis//2, kernel_size=5, stride=1),
            nn.BatchNorm1d(time_axis//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=3, stride=3),
            
            nn.Conv1d(time_axis//2, time_axis//4, kernel_size=5, stride=1),
            nn.BatchNorm1d(time_axis//4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(kernel_size=3, stride=3),
            
            nn.Conv1d(time_axis//4, time_axis//8, kernel_size=26, stride=1),
            nn.BatchNorm1d(time_axis//8),
            nn.ReLU(),
            nn.Dropout(0.3),
            )

        # Summary block
        self.summary = nn.Sequential(
            nn.Linear(59, no_classes),
            nn.ReLU(),
            nn.Dropout(0.3),
            )

    def forward(self, x):        
        #reshape to have frequency dim as channels:
        freq_first = x.permute(0, 2, 1, 3).squeeze()
        #print(freq_first.shape)
        freq_out = self.frequency(freq_first)
        #print(freq_out.shape)
        
        #reshape to have time dim as channels:
        time_first = x.permute(0, 3, 1, 2).squeeze()
        #print(time_first.shape)
        time_out = self.time(time_first)
        #print(time_out.shape)
        
        # Outer product:
        comb = torch.cat((freq_out, time_out), dim=1).squeeze()
        #print(comb.shape)
        
        out = self.summary(comb)
        #print(out.shape)

        return out

def test():
    image = torch.randn(64, 1, 256, 216)
    cnn = Puffin(256, 216, 10)
    output = cnn(image)
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    test()
