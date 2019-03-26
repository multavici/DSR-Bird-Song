#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:33:11 2019

@author: ssharma
"""
import torch
from torch.autograd import Variable
from .utils import printProgressBar
from time import perf_counter as pf
from .utils import profileit

@profileit('train_loop.prof')
def train(model, data_loader, epoch, optimizer, criterion, DEVICE):
    start = pf()
    model.train()
    model = model.to(DEVICE)

    n_correct = torch.FloatTensor([]).to(DEVICE)
    losses = torch.FloatTensor([]).to(DEVICE)
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        

        pred = output.data.max(1, keepdim=True)[1]
        corr = pred.eq(target.data.view_as(pred)).sum()
        n_correct = torch.cat((n_correct, corr.unsqueeze(dim=0).float()))
        losses = torch.cat((losses, loss.unsqueeze(dim=0).float()))

        latest_losses = losses[-50:]
        latest_correct = n_correct[-50:]

        running_loss = latest_losses.sum() / len(latest_losses)
        running_acc = latest_correct.sum() / (len(latest_correct) * data_loader.batch_size)

        printProgressBar(batch_idx + 1, 
                         len(data_loader),
                         prefix=f'Epoch: {epoch+1}',
                         suffix=f'Running Loss:{running_loss:.5f}, Running Acc:{running_acc:.5f}, Time: {pf()-start:.1f}',
                         length=50)
