#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:47:22 2019

@author: ssharma
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparrow(nn.Module):
    '''
    Model based on the sparrow SUBMISSION by Gril et.al 2017
    25th European Sugnal Processing Conference
    '''
           
    def __init__(self, time_axis=701, freq_axis=80,  no_classes=10):

        super(Sparrow , self).__init__()

        self.__name__='Sparrow'
                    
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, stride=1), #padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
           )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size=3, stride=1), #padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3)
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
            nn.MaxPool2d(kernel_size=3, stride =3)
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


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])

        return out

def main():
    image = torch.randn(1, 1, 80, 701)
    cnn = Sparrow (80, 701, 10)
    output = cnn(image)
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    main()
