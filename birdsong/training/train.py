#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:33:11 2019

@author: ssharma
"""

from torch.autograd import Variable
from .utils import printProgressBar


def train(model, data_loader, epoch, optimizer, criterion, DEVICE):
    model.train()
    model = model.to(DEVICE)
    
    n_correct = 0
    n_total = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        data = data.float()
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        n_total += len(target)
        running_loss += loss.item()

        current_loss = running_loss/(batch_idx+1)
        current_acc = n_correct/n_total

        printProgressBar(batch_idx + 1, len(data_loader),
                         prefix=f'Epoch: {epoch}',
                         suffix=f'Running Loss:{current_loss:.5f}, Running Acc:{current_acc:.5f}',
                         length=50)
