#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:01:00 2019

@author: tim
"""

from birdsong.datasets.tools.augmentation import *
from torchvision import transforms

MyTransform = transforms.Compose([VerticalRoll(10), GaussianNoise(0.1) ])


import torch
import numpy as np
img = np.arange(16).reshape(4,4)

MyTransform(img)


MyTransform.transforms