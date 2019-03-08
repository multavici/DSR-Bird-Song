#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:47:22 2019

@author: ssharma
"""

import numpy as np
import torch
import torch.nn as nn

class Sparrow(nn.Module):
    '''
    Model based on the sparrow SUBMISSION by Gril et.al 2017
    25th European Sugnal Processing Conference
    '''
    def __init__(self, freq_axis=701, time_axis=80, no_classes=10):

        super(Sparrow, self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.__name__ = 'Sparrow'

        self.freq_axis = np.floor_divide(self.freq_axis-4,3)
        self.freq_axis = np.floor_divide(self.freq_axis-22,3)

        self.time_axis = np.floor_divide(self.time_axis-4,3)
        self.time_axis = np.floor_divide(self.time_axis-6,3)
        self.time_axis = np.floor_divide(self.time_axis-8,1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, stride=1), #padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size=3, stride=1), #padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size=3, stride=1), # padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size=3, stride=1), # padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
           )

        self.layer5 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=(19,3), stride=1), # padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
           )

        self.layer6 = nn.Sequential(
            nn.Conv2d(64,256, kernel_size=(1,9), stride=1), # padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            )

        self.layer7 = nn.Sequential(
            nn.Conv2d(256,64, kernel_size=1, stride=1), # padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            )

        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1), # padding=1), 
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Dropout(0.5),
           )
        
        self.fc1 = nn.Linear(in_features=1 * self.time_axis * self.freq_axis, out_features=no_classes)   
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)

        return out

def main():
    image = torch.randn(1, 1, 80, 701)
    cnn = Sparrow(80, 701, 10)
    output = cnn(image)
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    main()
