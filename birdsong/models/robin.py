#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:46:03 2019

@author: ssharma
"""
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Robin(nn.Module):
    def __init__(self, freq_axis=256, time_axis=216,  no_classes=100):
        super(Robin, self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.no_classes = no_classes
        self.__name__='Robin'
        

        self.layer1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4,4),stride=(4,1)),
            nn.Conv2d(1,1, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3,5), stride =(1,5)),
            )

        self.layer2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4,4),stride=(1,4)),
            nn.Conv2d(1,1, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(4,3), stride =(4,1)),
            )

### 
        self.pool = nn.Sequential(
            nn.Conv2d(2,16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride =2),
            nn.Conv2d(16,1, kernel_size=(2, 2), stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ELU(),
            )

        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(800, 600),
                nn.ELU(),
                nn.Linear(600, 400),
                nn.ELU(),
                nn.Linear(400, 200),
                nn.ELU(),
                nn.Linear(200, self.no_classes),
                )
        

 

    def forward(self, x):
        out0 = self.layer1(x)
        out0 = F.pad(input=out0, pad=(3, 4, 0, 3), mode='constant', value=0)
        out1 = self.layer2(x)

        out2 = torch.cat((out0, out1),1)        
        out2 = self.pool(out2)
        out = self.fc(out2.reshape(out2.size(0), -1))

        return out
    

def test():
    image = torch.randn(5, 1, 256, 216)
    cnn = Robin(256, 216, 10)
    output = cnn(image)
    
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    test()
