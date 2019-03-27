#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:10:44 2019

@author: ssharma
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# A simple network from nn.Module
class Pigeon(nn.Module):
    def __init__(self, freq_axis=256, time_axis=216,  no_classes=10):
        super(Pigeon, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(32 * 64 * 54, no_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    
#img = torch.randn(5, 1, 256, 216)
#cnn = Pigeon(256, 216, 100)
#output = cnn(img)

