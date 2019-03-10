#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:06:04 2019

@author: ssharma
"""
import time
import os
import sys
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


import pandas as pd
from birdsong.training import train, evaluate, conf_mat, logger


sys.path.append("../birdsong")
from  FMnist_dataset import FashionMnist
    

def main(config_file):
     #read from config   
    local_config = __import__(config_file) 
    MODEL = local_config.inputs['MODEL']
    model = getattr(__import__('models.' + MODEL, fromlist=[MODEL]), MODEL) 
    batch_size = local_config.inputs['BATCHSIZE']
    optimizer = local_config.inputs['OPTIMIZER']
    num_epochs = local_config.inputs['EPOCHS']
    num_classes = local_config.inputs['CLASSES']
    lr = local_config.inputs['LR']
    
   
   #logging
    date = time.strftime('%d/%m/%Y')
    log_path = './run_log' + 'MODEL_' + date
    state_fname, log_fname, summ_tensor_board = logger.create_log(log_path)
    writer = SummaryWriter(summ_tensor_board)

    #ds_test = SpectralDataset(df_test)
    #ds_train = SpectralDataset(df_train)

    #dl_test = DataLoader(ds_test, BATCHSIZE)    S
    #dl_train = DataLoader(ds_train, BATCHSIZE)
    print('dataloaders initialized')
    

    #time_axis = ds_test.shape[1]
    #freq_axis = ds_test.shape[0]
    time_axis = 28 #goaway
    freq_axis = 28 #goaway

    dftrain = pd.read_csv('fashion-mnist_train.csv').sample(frac=0.1) #goaway
    dftest = pd.read_csv('fashion-mnist_test.csv').sample(frac=0.1) #goaway
    RESIZE = 28  #goaway
    transform_train = transforms.Compose([transforms.Resize(RESIZE), transforms.ToTensor()]) #goaway
    transform_test = transforms.Compose([transforms.Resize(RESIZE), transforms.ToTensor()]) #goaway
    criterion = nn.CrossEntropyLoss()
    
    fmnist_train = FashionMnist(dftrain, transform=transform_train) #goaway
    fmnist_test = FashionMnist(dftest, transform=transform_test) #goaway


    train_loader = DataLoader(fmnist_train, batch_size=batch_size) #goaway
    test_loader  = DataLoader(fmnist_test, batch_size=batch_size) #goaway


    
    
    net = model(time_axis=time_axis, freq_axis=freq_axis, num_classes=num_classes)
    optimizer = optim.Adam(net.parameters(), lr = lr)

    #local vars
    start_time = time.time()
    best_acc = 0
    for epoch in range(num_epochs):
        
        train(net, train_loader, epoch, optimizer, criterion, DEVICE)
        
        train_stats, train_conf_matrix = evaluate(train_loader, num_classes)
        test_stats, test_conf_matrix = evaluate(test_loader, num_classes)
        
        is_best = test_stats[1] > best_acc
        best_acc = max(test_stats[1], best_acc)
        print(best_acc)
        logger.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_accuracy': best_acc
        }, is_best, filename=state_fname)


        img = conf_mat.plot_conf_mat(test_conf_matrix) #new
        
        logger.write_summary(writer, epoch, train_stats, test_stats, img)
        
        logger.dump_log_txt(date, start_time, local_config, train_stats, test_stats, best_acc, log_fname)
                
                        

    writer.close()
    print('Finished Training')
    

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('usage: %s config_file' % os.path.basename(sys.argv[0]))
        sys.exit(2)

    config_file = os.path.basename(sys.argv[1])
 
    if config_file[-3:] == ".py":
        config_file = config_file[:-3]

    main(config_file)

