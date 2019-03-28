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


        # Harmony block
        self.harmony = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5,3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(2,3), stride=(2,3)),
            
            nn.Conv2d(16, 32, kernel_size=(5,3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(2,3), stride=(2,3)),
            
            nn.Conv2d(32, 64, kernel_size=(5,3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)),
            
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(1,7), stride=(1,7)),
            )
        
        # Next three run in parallel, picking up on harmonic features with different dilations
        self.summary1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,1), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            )
        
        self.summary2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,1), stride=1, dilation=2, padding=(1,0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            )
        
        self.summary3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,1), stride=1, dilation=3, padding=(2,0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1), stride=(3,1)),
            )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 51, no_classes),
            nn.ReLU()
            )
        
        
        

    def forward(self, x):         
        harmony_out = self.harmony(x)        
        summ1 = self.summary1(harmony_out)        
        summ2 = self.summary2(harmony_out)        
        summ3 = self.summary3(harmony_out)        
        comb = torch.cat((summ1,summ2,summ3), 2)
        comb = comb.view(comb.size(0), -1)
        out = self.fc(comb)        
        return out

def test():
    cnn = Puffin(256, 216, 100)
    summary(cnn, (1, 256, 216))

if __name__=="__main__":
    from torchsummary import summary
    test()
