
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

##########################################################################
# Get df of paths for pickled slices
df = pd.read_csv('slices_and_labels.csv')

def label_encoder(label_col):
    codes = {}
    i = 0
    for label in label_col.drop_duplicates():
        codes['label'] = i
        label_col[label_col == label] = i
        i += 1
    return label_col
df.label = label_encoder(df.label)

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
EPOCHS = 50
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
optimizer = optim.Adam(net.parameters(), lr = 0.001)


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



collect_metrics = []
collect_loss = []

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print("epoch", epoch)

    running_loss = 0.0
    for batch in dl_train:
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
    
    lltest, acctest = evaluate_model(net, dl_test)
    lltrain, acctrain = evaluate_model(net, dl_train)

    collect_metrics.append((lltest, lltrain, acctest, acctrain))
    print(f"----------EPOCH: {epoch} ----------")
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

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

y_test = []
y_pred = []
for i in range(len(ds_test)):
    x, y_true = ds_test[i]
    x = torch.Tensor(x).float().unsqueeze(dim = 0)
    y_p = net(x)
    y_test.append(y_true)
    y_pred.append(np.argmax(y_p.detach().numpy()))

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=list(range(CLASSES)), normalize=True,
                      title='Normalized confusion matrix')

plt.show()




