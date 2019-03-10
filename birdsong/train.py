#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:33:11 2019

@author: ssharma
"""

from torch.autograd import Variable
from utils import printProgressBar


def train(model, data_loader, epoch, optimizer, criterion, DEVICE):
    
    loss, n_correct, acc = 0., 0., 0.
    model.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        n_correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100. * n_correct / len(data)

        printProgressBar(batch_idx + 1, len(data_loader),
                         prefix=f'Epoch: {epoch}',
                         suffix=f'Batch Loss:{loss.item()} Batch Acc:{acc}',
                         length=50)
