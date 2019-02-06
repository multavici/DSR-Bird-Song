#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:07:07 2019

@author: tim
"""

"""
A prototype Dataset Class built around the Urban Sound dataset
tbc
"""

from torch.utils.data import Dataset
import PIL
import torchvision.transforms as transforms 

###############################################################################
# Dataset Class
class SpectralDataset(Dataset):
    def __init__(self, df):
        
        self.labels = df.label#.str.get_dummies()    #Possible cause no NaNs
        self.paths = df.path
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, i):
        #pdb.set_trace()
        path = self.paths.iloc[i]
        X = self.get_spectogram(path)
        y = self.labels.iloc[i]#.values
        return X, y
    
    def get_spectogram(self, path):
        slice_ = PIL.Image.open(path)
        return self.transform(slice_)
    
    def collate_func(self, batch):
        pass

