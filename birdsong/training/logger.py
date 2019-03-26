#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 09:05:09 2019

@author: ssharma
"""

import os
import json
import time
import torch
from pathlib import Path


def create_log(log_path):
    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    state_fname = os.path.join(log_dir, 'checkpoint.tar')
    log_fname = os.path.join(log_dir, 'run_log.log')
    summ_tensor_board = log_dir

    return state_fname, log_fname, summ_tensor_board


def write_summary(writer, epoch, train_stats, test_stats, img=None):
    # writer.add_image('Test/conf_mat', img, epoch, dataformats='HWC') #new
    writer.add_scalar('Train/loss', train_stats[0], epoch)
    writer.add_scalar('Train/acc', train_stats[1], epoch)
    writer.add_scalar('Train/top_5_acc', train_stats[2], epoch)
    writer.add_scalar('test/loss', test_stats[0], epoch)
    writer.add_scalar('test/acc', test_stats[1], epoch)
    writer.add_scalar('test/top_5_acc', test_stats[2], epoch)
    # writer.export_scalars_to_json("./all_scalars.json")


def save_checkpoint(state, is_best, filename='./checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")


def dump_log_txt(date, start_time, local_config, train_stats, test_stats, best_acc, epoch, log_fname):
    total_time = time.time() - start_time
    log = {
        'date': date,
        'no_classes': local_config.INPUTS['CLASSES'],
        'batchsize': local_config.INPUTS['BATCHSIZE'],
        'optimizer': local_config.INPUTS['OPTIMIZER'],
        'epochs': local_config.INPUTS['EPOCHS'],
        'learning_rate': local_config.INPUTS['LR'],
        'model': local_config.INPUTS['MODEL'],
        'final_epoch' : epoch,
        'best_accuracy_test' : best_acc,
        'final_accuracy_train': train_stats[1],
        'final_loss_train': train_stats[0],
        'final_accuracy_test': test_stats[1],
        'final_loss_test': test_stats[0],
        'total_time': total_time,
    }

    json.dump(log, open(log_fname, 'w'))
