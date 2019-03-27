#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:47:22 2019

@author: ssharma
"""


import torch
import torch.nn as nn
import numpy as np


class Bulbul(nn.Module):


    def __init__(self, freq_axis=80, time_axis=10000,  no_classes=10):

        super(Bulbul, self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.__name__ = 'Bulbul'

        for i in range(4):
            self.time_axis=np.floor_divide(self.time_axis-2,3)

        for i in range(2):
            self.freq_axis=np.floor_divide(self.freq_axis-2,3)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=3, stride=1), #padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3)
           )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size=3, stride=1), #padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3)
           )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size=(1,3), stride=1), # padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride =(1,3))
           )

        self.layer4 = nn.Sequential(
            nn.Conv2d(16,16, kernel_size=(1,3), stride=1), # padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3), stride =(1,3))
           )
        self.fc1 = nn.Linear(in_features=16 * self.time_axis * self.freq_axis, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=no_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.reshape(out.size(0), -1)  #reshape for fc
        out = self.fc1(out)
        out = self.fc2(out)
        return out



def test():
    image = torch.randn(64, 1, 256, 216)
    cnn = Bulbul(256, 216, 10)
    output = cnn(image)
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    test()

