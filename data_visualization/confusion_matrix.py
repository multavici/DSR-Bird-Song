from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import torch
import numpy as np

def plot_confusion_matrix(Dataset, model, CLASSES):
    y_test = []
    y_pred = []
    for i in range(len(Dataset)):
        x, y_true = Dataset[i]
        x = torch.Tensor(x).float().unsqueeze(dim = 0)
        y_p = model(x)
        y_test.append(y_true)
        y_pred.append(np.argmax(y_p.detach().numpy()))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    def _plot_confusion_matrix(cm, classes,
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
    _plot_confusion_matrix(cnf_matrix, classes=list(range(CLASSES)), normalize=True,
                          title='Normalized confusion matrix')
    plt.show()
