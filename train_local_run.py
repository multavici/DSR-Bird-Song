#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:38:54 2019

@author: ssharma
"""
"""
train on local machine :
using precomputed mel-spectrogram slices. It makes use of the static SpectralDataset
class to import files and relies upon a corresponding .csv for label metadata.
Make sure you have the slices present and unpacked into the storage directory!
"""

import json
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, log_loss

from torch.utils.data import DataLoader
from Datasets.static_dataset import SpectralDataset
from utils import printProgressBar
import os
import sys
import uuid

df_train = pd.read_csv('storage/df_train_local.csv')
df_test = pd.read_csv('storage/df_test_local.csv')
label_codes = pd.read_csv('storage/label_codes.csv')

##########################################################################

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(config_file):
    local_config = __import__(config_file) 
    MODEL = local_config.inputs['MODEL']
    model = getattr(__import__('models.'+ MODEL, fromlist=[MODEL]), MODEL)
    
    BATCHSIZE = local_config.inputs['BATCHSIZE']
    OPTIMIZER = local_config.inputs['OPTIMIZER']
    EPOCHS = local_config.inputs['EPOCHS']
    CLASSES = local_config.inputs['CLASSES']
    LR = local_config.inputs['LR']

    
    ##########################################################################
    
    ds_test = SpectralDataset(df_test)
    dl_test = DataLoader(ds_test, BATCHSIZE)
    
    ds_train = SpectralDataset(df_train)
    dl_train = DataLoader(ds_train, BATCHSIZE)
    print('dataloaders initialized')
       
    ##########################################################################
    
    time_axis = ds_test.shape[1]
    freq_axis = ds_test.shape[0]
    
    net = model(time_axis=time_axis, freq_axis=freq_axis, no_classes=CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = LR)
    
    # Keep only a single checkpoint, the best over test accuracy.
    def save_checkpoint(state, is_best, filename='./checkpoint.pth.tar'):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print ("=> Saving a new best")
            torch.save(state, filename)  # save checkpoint
        else:
            print ("=> Validation Accuracy did not improve")

    def evaluate_model(model, test_loader, print_info=False):
        with torch.no_grad():
            model.eval()
            collect_results = []
            collect_target = []
            for batch in test_loader:
                X, y = batch
                X = X.float()
    
                X = X.to(DEVICE)
                y = y.to(DEVICE).detach().cpu().numpy()
                pred = net(X)
    
                collect_results.append(F.softmax(pred, dim=-1).detach().cpu().numpy())
                collect_target.append(y) 
        
            preds_proba = np.concatenate(collect_results)
            preds = preds_proba.argmax(axis=1)
            
            targets = np.concatenate(collect_target)
            
            ll = log_loss(targets, preds_proba)
            acc = accuracy_score(targets, preds)
            if print_info:
                print("test log-loss: {}".format(ll))
                print("overall accuracy:  {}".format(acc))
                #print(classification_report(targets, preds))
            model.train()
            
            return ll, acc
        
    start_epoch = 0
    collect_loss = []
    best_acc = 0
    start_time = time.time()
    fname='checkpoint.'+str(uuid.uuid4())+'.tar'
    
    for epoch in range(EPOCHS):
        l = len(dl_train)
        printProgressBar(0, l, prefix = f'Epoch: {epoch}', suffix = 'Loss: 0', length = 50)
        for i, batch in enumerate(dl_train):
            X, y = batch
            X = X.float()
            
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            
            optimizer.zero_grad()

            y_pred = net(X)
            
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()  
            printProgressBar(i + 1, l, prefix = f'Epoch: {epoch}', suffix = f'Loss: {loss}', length = 50)

            #collect_loss.append(float(loss.detach().cpu().numpy()))  
            
        lltest, acctest = evaluate_model(net, dl_test)
        
        # collect_metrics.append((lltest, lltrain, acctest, acctrain))
        print("test: loss: {}  acc: {}".format(lltest, acctest))
        is_best = acctest > best_acc
        best_acc = max(acctest, best_acc)
        print(best_acc)
        
        save_checkpoint({
            'epoch': start_epoch + epoch + 1,
            'state_dict': net.state_dict(),
            'best_accuracy': best_acc
        }, is_best, filename=fname)
    
    
    print('Finished Training')
    total_time = time.time() - start_time
    
    log = {
        'date': time.strftime('%d/%m/%Y'),
        'no_classes': CLASSES,
        'batchsize': BATCHSIZE,
        'optimizer': OPTIMIZER,
        'epochs': EPOCHS,
        'learning_rate': LR,
        'model': net.__name__,
        'final_accuracy_test': acctest,
        'final_loss_test': lltest,
        'total_time': total_time,
        'check_point':fname
    
    }
    json.dump(log, open('/storage/runlog.txt', 'w+'))

    
if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('usage: %s config_file' % os.path.basename(sys.argv[0]))
        sys.exit(2)

    config_file = os.path.basename(sys.argv[1])
    if config_file[-3:] == ".py":
        config_file = config_file[:-3]

    local_config = __import__(config_file, globals(), locals(), [])

    main(config_file)









