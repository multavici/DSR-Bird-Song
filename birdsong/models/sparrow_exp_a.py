#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:24:26 2019

@author: ssharma
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.custom_layers.exp_function import ExpFunctionLayer
except:
    from .custom_layers.exp_function import ExpFunctionLayer
    
class SparrowExpA(nn.Module):
    '''
    Model based on the sparrow varian A by Jan Schlüter et.al 2018
    '''

    def __init__(self, time_axis=701, freq_axis=80,  no_classes=10):
        super(SparrowExpA , self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.__name__='SparrowExpA'

        self.layer0 = ExpFunctionLayer(1,1, bias=None)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=3, stride=1), #padding=1),
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, stride=1), #padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3)
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, stride=1), #padding=1),
            nn.ReLU(),
            )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, stride=1), #padding=1),
            nn.ReLU(),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=(17,3), stride=1), #padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,3), stride =3),
            )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128,1024, kernel_size=(1,9), stride=1), #padding=1),
            nn.ReLU(),
            )

        self.layer7 = nn.Sequential(
            nn.Conv2d(1024, no_classes, kernel_size=(1,1), stride=1), #padding=1),
            nn.ReLU(),
            )

    def forward(self, x):
        out = x.reshape(x.numel(),1)
        out = self.layer0(out)
        out = out.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = F.max_pool2d(out, kernel_size=out.size()[2:]).view(out.size()[0], out.size()[1])
        return out


def main():
    image = torch.randn(1, 1, 128, 212)
    cnn = SparrowExpA (128, 212, 10)
    output = cnn(image)
    print("input shape:")
    print(image.shape)
    print("output shape:")
    print(output.shape)

if __name__ == '__main__':
    main()
