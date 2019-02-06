#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:10:16 2019

@author: tim
"""
import numpy as np
import pandas as pd
from pytorch_spectral_dataset import SpectralDataset
from torch import nn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, log_loss
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load csv here
df = pd.read_csv('spectral_slices.csv')

def label_encoder(df):
    codes = {}
    i = 0
    for label in df.label.drop_duplicates():
        codes['label'] = i
        df.label[df.label == label] = i
        i += 1
    return df

df = label_encoder(df)    

# Get even class representation
sample_size = 400
df = df.groupby('label').apply(lambda x: x.sample(sample_size))

#Split in test and train:
indeces = np.random.permutation(len(df))
split = int(len(indeces) * 0.8)
train_indeces = indeces[:split]
test_indeces = indeces[split:]


train = df.iloc[train_indeces]
test = df.iloc[test_indeces]


classes = 4

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
    
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(409600, classes)

    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)

        out = self.fc(out)
        return out
    
# Always check your model are you atleasy able to make a forward pass and shapes match your expectations?
image = torch.randn(1, 1, 1025, 200)
cnn = SimpleCNN()
output = cnn(image)
print("input shape:")
print(image.shape)
print("output shape:")
print(output.shape)


LR= 0.001
BATCH_SIZE = 64


cnn = SimpleCNN()

OPTIMIZER = 'Adam' # one of ['ASGD','Adadelta', 'Adagrad','Adam', 'Adamax','LBFGS', 'RMSprop','Rprop','SGD',SparseAdam']
optimizer = getattr(torch.optim, OPTIMIZER)(cnn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

cnn.to(DEVICE)

train_ds = SpectralDataset(train)
test_ds = SpectralDataset(test)

# Create dataset loader
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Lets try to use the criterion with dummy data
yp = torch.randn(BATCH_SIZE, 4)
yt = torch.randint(4, (BATCH_SIZE,))
criterion(yp, yt.long())



###############################################################################
def evaluate_model(model, test_loader, print_info=False):
    with torch.no_grad():
        model.eval()
        collect_results = []
        collect_target = []
        for batch in test_loader:
            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE).detach().cpu().numpy()
            pred = cnn(X)
            collect_results.append(F.softmax(pred).detach().cpu().numpy())
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
print('Begin training...')
for epoch in range(20):
    lossacc = 0
    for i, batch in enumerate(train_dl):
        print(f'Batch number {i}')
        optimizer.zero_grad()
        X, y = batch
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        y_pred = cnn(X)
        
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()  
        collect_loss.append(float(loss.detach().cpu().numpy()))  
        
    lltest, acctest = evaluate_model(cnn, test_dl)
    lltrain, acctrain = evaluate_model(cnn, train_dl)
    collect_metrics.append([lltest, lltrain, acctest, acctrain])
    print("test: loss: {}  acc: {}".format(lltest, acctest))
    print("train: loss: {}  acc: {}".format(lltrain, acctrain))