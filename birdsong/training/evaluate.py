
import numpy as np
import torch
from torch.autograd import Variable

from .conf_mat import calc_conf_mat
from .utils import top_k_accuracy


def evaluate(model, data_loader, criterion, num_classes, DEVICE):
    model.eval()
    model = model.to(DEVICE)

    n_correct = 0
    loss = 0
    top_5 = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)
            data = data.float()
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            print('target', target)

            output = model(data)

            loss += criterion(output, target).item()

            _, topk = output.topk(5, dim=1)
            print('ouput.topk', topk)
            #_, pred = output.max(1, keepdim=True)
            #print('pred', pred)

            print('torch.eq', torch.eq(topk, target))

            top_5_batch = top_k_accuracy(output, target, topk=(5,))
            top_5.append(top_5_batch)

            pred = output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            if batch_idx == 0:
                pred_cat = pred
                targ_cat = target
            else:
                pred_cat = torch.cat((pred_cat, pred))
                targ_cat = torch.cat((targ_cat, target))

    conf_matrix = calc_conf_mat(pred_cat, targ_cat, num_classes)

    loss /= len(data_loader)
    acc = n_correct / len(data_loader.dataset)
    top_5_acc = np.mean(top_5)

    return (loss, acc, top_5_acc), conf_matrix
