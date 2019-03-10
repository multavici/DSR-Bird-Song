#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:21:30 2019

@author: ssharma
"""
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def calc_conf_mat(pred, target, num_classes):

    predicted = pred.squeeze(dim = -1).cpu().numpy()
    target = target.cpu().numpy()

    assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

    x = predicted + num_classes * target
    bincount_2d = np.bincount(x.astype(np.int32),
                                      minlength=num_classes ** 2)
    conf_mat = bincount_2d.reshape((num_classes, num_classes))
        
    return conf_mat


def plot_conf_mat(conf_matrix):
    
    fig = plt.figure(figsize=(12,12))
    
    normalize = 1
    if normalize:
        cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap='Blues')

    else:
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')


   # print(cm)

    plt.title('Confusion Matrix')
    plt.colorbar()

   # fmt = '.2f' if normalize else 'd'
   # thresh = cm.max() / 2.
    
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)

    buf_img = buf.read()

    image = np.array(Image.open(BytesIO(buf_img))).astype(np.uint8)
    img = image[:, :, :3]
    
    return img
