
import numpy as np
import torch
from torch.autograd import Variable
#from sklearn.metrics import confusion_matrix
from .conf_mat import plot_confusion_matrix
from .utils import profileit

@profileit('evaluation_loop.prof')
def evaluate(model, data_loader, criterion, num_classes, DEVICE):
    model.eval()
    model = model.to(DEVICE)

    n_correct = 0
    n_intopk = 0
    loss = 0

    k = 5

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)

            loss += criterion(output, target).item()

            topk = output.topk(k, dim=1)[1]
            pred = output.data.max(1, keepdim=True)[1]

            n_correct += pred.eq(target.data.view_as(pred)).sum().item()
            n_intopk += topk.eq(target.reshape(len(target), 1)
                                ).sum().item()

            if batch_idx == 0:
                pred_cat = pred
                targ_cat = target
            else:
                pred_cat = torch.cat((pred_cat, pred))
                targ_cat = torch.cat((targ_cat, target))

    loss /= len(data_loader)
    acc = n_correct / len(data_loader.dataset)
    top_5_acc = n_intopk / len(data_loader.dataset)

    return (loss, acc, top_5_acc), (targ_cat.cpu(), pred_cat.cpu())
