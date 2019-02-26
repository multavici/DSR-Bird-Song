
# Test Run
import pandas as pd
import numpy as np
from Spectrogram.spectrograms import mel_s, stft_s
from torch.utils.data import DataLoader
from Datasets.static_dataset import SoundDataset
from get_chunks import get_records_from_classes
from models.bulbul import BulBul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, log_loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


##########################################################################
class_ids = [6088, 3912, 4397, 7091, 4876, 4873, 5477, 6265, 4837, 4506] # all have at least 29604 s of signal, originally 5096, 4996, 4993, 4990, 4980
df = get_records_from_classes(class_ids, 100)
print('df created')

def label_encoder(label_col):
    codes = {}
    i = 0
    for label in label_col.drop_duplicates():
        codes['label'] = i
        label_col[label_col == label] = i
        i += 1
    return label_col
df.label = label_encoder(df.label)

# Check sample distribution:
df.groupby('label').agg({'total_signal':'sum'})

# Split into train and test
msk = np.random.rand(len(df)) < 0.8
df_train = df.iloc[msk]
df_test = df.iloc[~msk]
print('train and test dataframes created')
##########################################################################





##########################################################################
BATCHSIZE = 64
OPTIMIZER = 'Adam'
EPOCHS = 10

# Parameters for sample loading
params = {'batchsize' : BATCHSIZE, 
          'window' : 5000, 
          'stride' : 1000, 
          'spectrogram_func' : stft_s, 
          'augmentation_func' : None}


ds_test = SoundDataset(df_test, **params)
dl_test = DataLoader(ds_test, BATCHSIZE)

ds_train = SoundDataset(df_train, **params)
dl_train = DataLoader(ds_train, BATCHSIZE)





##########################################################################
time_axis = ds_test[0][0].shape[1]
freq_axis = ds_test[0][0].shape[0]

net = BulBul(time_axis, freq_axis, len(class_ids))


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(Bulbul.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr = 0.0001)


def evaluate_model(model, test_loader, print_info=False):
    with torch.no_grad():
        model.eval()
        collect_results = []
        collect_target = []
        for batch in test_loader:
            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE).detach().cpu().numpy()
            pred = net(X)

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

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dl_train):
        # get the inputs
        X, y = data
            
        X = X.to(DEVICE)
        y = y.to(DEVICE)

 
        # zero the parameter gradients
        optimizer.zero_grad()

        y_pred = net(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    
    lltest, acctest = evaluate_model(net, dl_test)
    lltrain, acctrain = evaluate_model(net, dl_train)

    collect_metrics.append((lltest, lltrain, acctest, acctrain))
    print("test: loss: {}  acc: {}".format(lltest, acctest))
    print("train: loss: {}  acc: {}".format(lltrain, acctrain))

print('Finished Training')

    

