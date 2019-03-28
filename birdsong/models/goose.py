#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:47:22 2019

@author: ssharma
"""


import torch
import torch.nn as nn
import numpy as np


class Goose(nn.Module):


    def __init__(self, freq_axis=80, time_axis=10000,  no_classes=10):

        super(Goose, self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.no_classes = no_classes
        self.__name__ = 'Goose'


        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=(3,1), stride=(3,1)), #padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1), stride =(3,1))
           )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size=(3,1), stride=(3,1)), #padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,1), stride =(3,1))
           )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16,1, kernel_size=(2,1), stride=1), # padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride =1)
           )

 
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(256,1), stride =(256,1))
           )
       
    #after concat
        self.pool1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=(2,2), stride=(1,1), padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride =(2,2)),
            )
       
        self.fc = nn.Sequential(
        #        nn.Dropout(0.5),
                nn.Linear(1728, 864),
                nn.ELU(),
                nn.Linear(864, 432),
                nn.ELU(),
                nn.Linear(432, self.no_classes),
                )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out_t = self.layer4(x)
        #concat
        pool = torch.cat((out, out_t),2)
        
        pool = self.pool1(pool)
        out = self.fc(pool.reshape(pool.size(0), -1))
        return out



def test():
    image = torch.randn(5, 1, 256, 216)
    cnn = Goose(256, 216, 10)
    output = cnn(image)
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    test()
