
# Test Run
import sqlite3
import pandas as pd
import numpy as np
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, log_loss

from torch.utils.data import DataLoader
from Datasets.static_dataset import SpectralDataset
from models.bulbul import Bulbul
import os

##########################################################################
# Get df of paths for pickled slices
def get_df(): 
    conn = sqlite3.connect('storage/db.sqlite')
    c = conn.cursor()
    def lookup(id):
        c.execute("""SELECT r.taxonomy_id FROM recordings
            WHERE r.id = ?""", (id,))
        tax_id = c.fetchone()[0]
        return tax_id
    list_recs = []
    for dirpath, dirname, filenames in os.walk('storage/slices'):
        for name in filenames:
            path = os.path.join(dirpath, name)
            id = dirpath.split("/")[2]
            species = lookup(id)
            list_recs.append((str(path), species))   
    df = pd.DataFrame(list_recs, columns=['path', 'label'])
    return df

df = get_df()

for _, row in df.iterrows():
    row['path'].split("/")[2]


# Split into train and test
msk = np.random.rand(len(df)) < 0.8
df_train = df.iloc[msk]
df_test = df.iloc[~msk]
print('train and test dataframes created')



##########################################################################

start_time = time.time()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initiate graph for paperspace
print('{"chart": "train accuracy", "axis": "epochs"}')
print('{"chart": "test accuracy", "axis": "epochs"}')


##########################################################################

BATCHSIZE = 32
OPTIMIZER = 'Adam'
EPOCHS = 10
CLASSES = 10


##########################################################################

ds_test = SpectralDataset(df_test)
dl_test = DataLoader(ds_test, BATCHSIZE)

ds_train = SpectralDataset(df_train)
dl_train = DataLoader(ds_train, BATCHSIZE)
print('dataloaders initialized')


##########################################################################

time_axis = ds_test[0][0].shape[2]
freq_axis = ds_test[0][0].shape[1]

net = Bulbul(time_axis=time_axis, freq_axis=freq_axis, no_classes=CLASSES)


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

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print("epoch", epoch)

    running_loss = 0.0
    for batch in dl_train:
        print("batch")
        # get the inputs
        X, y = batch
        print("X", X.shape)
        print("y", y.shape)
        
        X = X.float()
        
        X = X.to(DEVICE)
        y = y.to(DEVICE)

 
        # zero the parameter gradients
        optimizer.zero_grad()

        y_pred = net(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print("batch finished")
    
    lltest, acctest = evaluate_model(net, dl_test)
    lltrain, acctrain = evaluate_model(net, dl_train)

    collect_metrics.append((lltest, lltrain, acctest, acctrain))
    print(f"----------EPOCH: {epoch} ----------")
    print("test: loss: {}  acc: {}".format(lltest, acctest))
    print("train: loss: {}  acc: {}".format(lltrain, acctrain))
    print('{"chart": "train accuracy", "x": {}, "y": {}}'.format(epoch, acctrain))
    print('{"chart": "test accuracy", "x": {}, "y": {}}'.format(epoch, acctest))

print('Finished Training')
total_time = time.time() - start_time

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

