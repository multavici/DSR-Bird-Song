#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:33:11 2019

@author: ssharma
"""

from torch.autograd import Variable
from .utils import printProgressBar
from time import perf_counter as pf
from .utils import profileit

@profileit('train_loop.prof')
def train(model, data_loader, epoch, optimizer, criterion, DEVICE):
    start = pf()
    model.train()
    model = model.to(DEVICE)

    n_correct = []
    losses = []
    printProgressBar(0, 
                     len(data_loader),
                     prefix=f'Epoch: {epoch+1}',
                     suffix=f'Running Loss: 0, Running Acc: 0, Time: {pf()-start:.1f}',
                     length=50)

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
        n_correct.append(pred.eq(target.data.view_as(pred)).cpu().sum().item())
        losses.append(loss.item())

        latest_losses = losses[-50:]
        latest_correct = n_correct[-50:]

        running_loss = sum(latest_losses) / len(latest_losses)
        running_acc = sum(latest_correct) / (len(latest_correct) * data_loader.batch_size)

        if batch_idx % 5 == 0 or batch_idx == len(data_loader):
            printProgressBar(batch_idx + 1, 
                             len(data_loader),
                             prefix=f'Epoch: {epoch+1}',
                             suffix=f'Running Loss:{running_loss:.5f}, Running Acc:{running_acc:.5f}, Time: {pf()-start:.1f}',
                             length=50)
