import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #percentage: 
        cm = cm.astype('float') * 100
        # add percentage sign

        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    mycm = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    mycm.set_clim([0,100])
    plt.title(title)
    cbar = plt.colorbar(mycm, shrink=0.72, ticks=list(range(0, 120, 20)))
    cbar.ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])  # vertically oriented colorbar

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,  ha="right")
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(format(cm[i, j], fmt)) + "%", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')