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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Owl(nn.Module):
    def __init__(self, freq_axis=256, time_axis=216,  no_classes=100):
        super(Owl, self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.no_classes = no_classes
        self.y_axis = torch.div(freq_axis, 2.)
        self.__name__='Owl'
        
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4,4),stride=(4,1)),
            nn.Conv2d(1,1, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,5), stride =(1,5)),
            )

        self.pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4,4),stride=(1,4)),
            nn.Conv2d(1,1, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,3), stride =(4,1)),
            )
        '''
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4,1),stride=(4,1)),
            nn.Conv2d(1,1, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3),
            )

        self.pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,6),stride=(1,6)),
            nn.Conv2d(1,1, kernel_size=(4, 4), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride =3),
            )
        '''
        self.pool3 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride =2),

            nn.Conv2d(16,32, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride =2),

            nn.Conv2d(32,16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride =2),
            )
        
###
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(672, 336),
                nn.BatchNorm1d(336),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(336, self.no_classes),
                nn.Sigmoid()                
                )


    def forward(self, x):
        out0 = self.pool1(x)
        out0 = F.pad(input=out0, pad=(3, 4, 0, 3), mode='constant', value=0)

        out1 = self.pool2(x)
        out2 = torch.add(out0,1,out1)
        
        out2 = self.pool3(out2)

        out = self.fc(out2.reshape(out2.size(0), -1))
        return out
    

#image = unpack('../storage/step1_slices/47314_10.pkl')
#img = to_ten(image)
#img = torch.randn(5, 1, 256, 216)
#cnn = Eagle(256, 216, 100)
#outputf, outputt = cnn(img)
#outf_img=outputf.squeeze(0).squeeze(0).detach().numpy()
#outt_img=outputt.squeeze(0).squeeze(0).detach().numpy()
#output = cnn(img)
