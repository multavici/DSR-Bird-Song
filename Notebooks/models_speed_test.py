#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:58:35 2019

@author: tim
"""

# Speed test models:

import torch
from torch import nn
import torch.optim as optim
import numpy as np
from time import perf_counter as pf
from birdsong.models import Bulbul, Sparrow, SparrowExpA, Zilpzalp

freq = 128
time = 212
batch_size = 32

Bul = Bulbul(freq, time, 10)
Spa = Sparrow(freq, time, 10)
Spax = SparrowExpA(freq, time, 10)
Zil = Zilpzalp(freq, time, 10)

models = [Bul, Spa, Spax, Zil]

test_img = torch.randn(batch_size, 1, freq, time)
target = torch.Tensor(np.array([0] * batch_size)).long()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


test_img = test_img.to(DEVICE)
target = target.to(DEVICE)

criterion = nn.CrossEntropyLoss()


for model in models:
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start = pf()
    optimizer.zero_grad()
    output = model(test_img)
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()
    print(model.__name__, ': ', (pf()-start), ' s')
