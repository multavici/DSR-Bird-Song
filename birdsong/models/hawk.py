#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:43:10 2019

@author: ssharma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class Hawk(nn.Module):
    def __init__(self, freq_axis=256, time_axis=216,  no_classes=10):
        super(Hawk, self).__init__()

        self.time_axis = time_axis
        self.freq_axis = freq_axis
        self.no_classes = no_classes
        self.y_axis = torch.div(freq_axis, 2.)
        self.__name__='Hawk'
        
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2,1),stride=(2,1)),
            )        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=(7, int(0.90*self.y_axis)), stride=1), #padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,102), stride =(1,102))
           )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(3, int(0.90*self.y_axis)), stride=1), #padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,102), stride =(1,102))
           )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=(1, int(0.90*self.y_axis)), stride=1), #padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,102), stride =(1,102))
           )
###
        self.conv4 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=(7, int(0.40*self.y_axis)), stride=1), #padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,166), stride =(1,166))
           )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=(3, int(0.40*self.y_axis)), stride=1), #padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,166), stride =(1,166))
           )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=(1, int(0.40*self.y_axis)), stride=1), #padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,166), stride =(1,166))
           )
###

        self.pool7 = nn.Sequential(
                nn.AvgPool2d(kernel_size=(1,216),stride=(1,216)),
                )

        self.conv7 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=165, stride=1, padding=82),
                nn.BatchNorm1d(16), 
                nn.ReLU(),
           )

        self.conv8 = nn.Sequential(                
                nn.Conv1d(1,32, kernel_size=125, stride=1, padding=62),
                nn.BatchNorm1d(32), 
                nn.ReLU(),
           )

        self.conv9 = nn.Sequential(                
                nn.Conv1d(1,64, kernel_size=65, stride=1, padding=32),
                nn.BatchNorm1d(64), 
                nn.ReLU(),
           )

        self.conv10 = nn.Sequential(                
                nn.Conv1d(1,128, kernel_size=31, stride=1, padding=15),
                nn.BatchNorm1d(128), 
                nn.ReLU(),
           )

###
        
        self.conv_bn1 = nn.Sequential(                
                nn.Conv2d(1,512, kernel_size=(464,7), stride=1),
                nn.BatchNorm2d(512), 
                nn.ReLU(),
           )
        self.conv_bn2 = nn.Sequential(                
                nn.Conv2d(1,512, kernel_size=(7,512), stride=1),
                nn.BatchNorm2d(512), 
                nn.ReLU(),
           )
        #temporal pooling    
        self.pool_t = nn.Sequential(
                nn.MaxPool2d(kernel_size=(2,1), stride =(2,1))
                )
        self.conv11 = nn.Sequential(                
                nn.Conv2d(1,512, kernel_size=(7,512), stride=1),
#                nn.ReLU(),
           )
        self.conv11_b = nn.Sequential(                
                nn.BatchNorm2d(1),
                nn.ReLU(),
           )
        
###
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, self.no_classes),
                nn.Sigmoid()                
                )

        
    def forward(self, x):
        out0 = self.pool1(x)
        input_pad_7 = F.pad(input=out0, pad=(0, 0, 3, 3), mode='constant', value=0)
        input_pad_3 = F.pad(input=out0, pad=(0, 0, 1, 1), mode='constant', value=0)

        out1 = self.conv1(input_pad_7)
        out1 = out1.squeeze(3)
        out2 = self.conv2(input_pad_3)
        out2 = out2.squeeze(3)
        out3 = self.conv3(out0)
        out3 = out3.squeeze(3)

        out4 = self.conv4(input_pad_7)
        out4 = out4.squeeze(3)
        out5 = self.conv5(input_pad_3)
        out5 = out5.squeeze(3)
        out6 = self.conv6(out0)
        out6 = out6.squeeze(3)
        
        p7 = self.pool7(out0)
        p7 = p7.squeeze(3)  
        
        out7 = self.conv7(p7)
        out8 = self.conv8(p7)
        out9 = self.conv9(p7)
        out10 = self.conv10(p7)
       
        pool = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8, out9, out10), 1)
        pool = pool.unsqueeze(1)  #.permute(0,1,3,2)
        
        bn1 = self.conv_bn1(pool)
        bn1 = bn1.permute(0,2,3,1)
        bn1_pad = F.pad(input=bn1, pad=(0, 0, 3, 3), mode='constant', value=0)
        bn2 = self.conv_bn2(bn1_pad)
        bn2 = bn2.permute(0,3,2,1)
        
        bn_12 = bn1.add(bn2)

        pool_t = self.pool_t(bn_12)
        pool_tpad = F.pad(input=pool_t, pad=(0, 0, 3, 3), mode='constant', value=0)        
        out11 = self.conv11(pool_tpad)
        out11 = out11.permute(0,3,2,1)
        out11 = self.conv11_b(out11)
        out12 = out11.add(pool_t)
        max_pool = torch.max(out12, 2)[0]
        mean_pool = torch.mean(out12, 2)
        pool2 = torch.cat((max_pool, mean_pool), 1)
                 
        out = self.fc(pool2.reshape(pool2.size(0), -1))
        
        return out






#image = torch.randn(5, 1, 256, 216)
#cnn = Hawk(256, 216, 10)
#output = cnn(image)

