#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:15:19 2019

@author: tim
"""


# Test Run
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, log_loss

from torch.utils.data import DataLoader
from Datasets.dynamic_dataset import SoundDataset
from models.sparrow import Sparrow
from Spectrogram.spectrograms import mel_s
from data_preparation.utils import get_downloaded_records_from_classes
#from models.bulbul import Bulbul

##########################################################################
class_ids = []
df = get_downloaded_records_from_classes(class_ids, seconds_per_class = 200, min_signal_per_file = 3000)

def label_encoder(label_col):
    codes = {}
    i = 0
    for label in label_col.drop_duplicates():
        codes['label'] = i
        label_col[label_col == label] = i
        i += 1
    return label_col
df.label = label_encoder(df.label)

#df = df.groupby('label').head(1000)

#sample_size = df.groupby('label').count().min().values[0]
#df = df.reset_index(drop = True).groupby('label').apply(lambda x: x.sample(sample_size)).reset_index(drop = True)

# Split into train and test
msk = np.random.rand(len(df)) < 0.8
df_train = df.iloc[msk]
df_test = df.iloc[~msk]
print('train and test dataframes created')

##########################################################################

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initiate graph for paperspace
print('{"chart": "train accuracy", "axis": "epochs"}')
print('{"chart": "test accuracy", "axis": "epochs"}')


##########################################################################

BATCHSIZE = 32
OPTIMIZER = 'Adam'
EPOCHS = 50
CLASSES = 10

params = {'batchsize' : 32, 
          'window' : 3000, 
          'stride' : 1500, 
          'spectrogram_func' : mel_s, 
          'augmentation_func' : None}

##########################################################################

ds_test = SoundDataset(df_test, **params)
dl_test = DataLoader(ds_test, BATCHSIZE)

ds_train = SoundDataset(df_train, **params)
dl_train = DataLoader(ds_train, BATCHSIZE)
print('dataloaders initialized')


##########################################################################

time_axis = ds_test.shape[1]
freq_axis = ds_test.shape[0]

net = Sparrow(time_axis=time_axis, freq_axis=freq_axis, no_classes=CLASSES)
#net = Sparrow(time_axis=time_axis, freq_axis=freq_axis, no_classes=CLASSES)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(Bulbul.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr = 0.001)


def evaluate_model(model, test_loader, print_info=False):
    with torch.no_grad():
        model.eval()
        collect_results = []
        collect_target = []
        for i, batch in enumerate(test_loader):
            print(f'Batch {i+1}')
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



collect_metrics = []
collect_loss = []


start_time = time.time()
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print("epoch", epoch)

    running_loss = 0.0
    for i, batch in enumerate(dl_train):
        # get the inputs
        X, y = batch

        X = X.float()
        
        X = X.to(DEVICE)
        y = y.to(DEVICE)

 
        # zero the parameter gradients
        optimizer.zero_grad()

        y_pred = net(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f'Batch {i+1}, Loss: {loss}')

    
    lltest, acctest = evaluate_model(net, dl_test)
    lltrain, acctrain = evaluate_model(net, dl_train)

    collect_metrics.append((lltest, lltrain, acctest, acctrain))
    print(f"----------EPOCH: {epoch} ----------")
    print("time: ", time.time() - start_time)
    print("test: loss: {}  acc: {}".format(lltest, acctest))
    print("train: loss: {}  acc: {}".format(lltrain, acctrain))
    #print('{"chart": "train accuracy", "x": {}, "y": {}}'.format(epoch, acctrain))
    #print('{"chart": "test accuracy", "x": {}, "y": {}}'.format(epoch, acctest))

print('Finished Training')
total_time = time.time() - start_time
"""
log = {
    'date': time.strftime('%d/%m/%Y'),
    'no_classes': CLASSES,
    'batchsize': BATCHSIZE,
    'optimizer': OPTIMIZER,
    'epochs': EPOCHS,
    'model': net.__name__,
    'final_accuracy_test': acctest,
    'final_accuracy_train': acctrain,
    'final_loss_test': lltest,
    'final_loss_train': lltrain,
    'total_time': total_time,

}
json.dump(log, open('/storage/runlog.txt', 'w+'))
"""






