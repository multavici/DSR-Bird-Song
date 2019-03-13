#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:06:04 2019

@author: ssharma
"""
import time
import os
import sys
sys.path.append("./birdsong")
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
from tensorboardX import SummaryWriter
from training import train, evaluate, logger, plot_conf_mat
from datasets.sequential import RandomSpectralDataset, SpectralDataset
from datasets.tools.enhancement import exponent

if 'HOSTNAME' in os.environ:
    # script runs on server
    INPUT_DIR = '/storage/step1_slices/'
else:
    # script runs locally
    INPUT_DIR = 'storage/slices/'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
TRAIN = pd.read_csv('mel_slices_train.csv')
TEST = pd.read_csv('mel_slices_test.csv')

def main(config_file):
    #read from config
    local_config = __import__(config_file)
    model_name = local_config.INPUTS['MODEL']
    model = getattr(__import__('.models', fromlist=[model_name]), model_name)
    batch_size = local_config.INPUTS['BATCHSIZE']
    optimizer = local_config.INPUTS['OPTIMIZER']
    num_epochs = local_config.INPUTS['EPOCHS']
    no_classes = local_config.INPUTS['CLASSES']
    learning_rate = local_config.INPUTS['LR']

    #logging
    start_time = time.time()
    date = time.strftime('%d-%m-%Y-%H-%M-%S', time.localtime())
    log_path = f'./birdsong/run_log/{model_name}_{date}'
    state_fname, log_fname, summ_tensor_board = logger.create_log(log_path)
    writer = SummaryWriter(str(summ_tensor_board))
    
    ds_test = RandomSpectralDataset(TEST, INPUT_DIR, slices_per_class= 400, examples_per_batch=3, enhancement_func=exponent)
    ds_train = SpectralDataset(TRAIN, INPUT_DIR, enhancement_func=exponent)
    dl_test = DataLoader(ds_test, batch_size)
    dl_train = DataLoader(ds_train, batch_size)
    print('dataloaders initialized')

    time_axis = ds_test.shape[1]
    freq_axis = ds_test.shape[0]
    net = model(time_axis=time_axis, freq_axis=freq_axis, no_classes=no_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    #local vars
    best_acc = 0
    for epoch in range(num_epochs):
        train(net, dl_train, epoch, optimizer, criterion, DEVICE)

        train_stats, train_conf_matrix = evaluate(net, dl_train, criterion, no_classes, DEVICE)
        print(f'Train Loss: {train_stats[0]:.5f}, Train Acc: {train_stats[1]:.5f}')
        test_stats, test_conf_matrix = evaluate(net, dl_test, criterion, no_classes, DEVICE)
        print(f'Test Loss: {test_stats[0]:.5f}, Test Acc: {test_stats[1]:.5f}')

        is_best = test_stats[1] > best_acc
        best_acc = max(test_stats[1], best_acc)
        print('Best Accuracy: {:.5f}'.format(best_acc))

        logger.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_accuracy': best_acc
        }, is_best, filename=state_fname)

        img = plot_conf_mat(test_conf_matrix) #TODO: Pass train_conf_matrix as well
        logger.write_summary(writer, epoch, train_stats, test_stats, img)
        logger.dump_log_txt(date, start_time, local_config, train_stats, test_stats, best_acc, log_fname)

    writer.close()
    print('Finished Training')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('usage: %s config_file' % os.path.basename(sys.argv[0]))
        sys.exit(2)

    CONFIG = os.path.basename(sys.argv[1])

    if CONFIG[-3:] == ".py":
        CONFIG = CONFIG[:-3]

    main(CONFIG)
