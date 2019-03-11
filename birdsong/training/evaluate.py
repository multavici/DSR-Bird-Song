#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:10:53 2019

@author: ssharma
"""
#import numpy as np
import torch
from torch.autograd import Variable
#from sklearn import metrics
from .conf_mat import calc_conf_mat


def evaluate(model, data_loader, criterion, num_classes, DEVICE):

    model.eval()
    model = model.to(DEVICE)
    loss = 0
    correct = 0
#    conf = np.ndarray((num_classes, num_classes))

    i = 0
    with torch.no_grad():
        for data, target in data_loader:
            
            data, target = Variable(data), Variable(target)
            data = data.float()
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)
            
            #loss += F.cross_entropy(output, target, reduction='sum').data.item()   
            loss +=criterion(output,target)
            
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            if i == 0:
                pred_cat = pred
                targ_cat = target
                i = 1
            else:
                pred_cat = torch.cat((pred_cat, pred))
                targ_cat = torch.cat((targ_cat, target))

    conf_matrix = calc_conf_mat(pred_cat, targ_cat, num_classes)
    #from sklearn   
    #conf_matrix = metrics.confusion_matrix(pred_cat.view(-1), targ_cat.view(-1)) 
    
    loss /= len(data_loader.dataset)

    acc = 100. * correct / len(data_loader.dataset)
        
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
        loss, correct, len(data_loader.dataset), acc))

    return (loss.item(), acc), conf_matrix
