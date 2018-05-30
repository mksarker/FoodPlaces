"""
@author: Md Mostafa Kamal Sarker
@ Department of Computer Engineering and Mathematics, Universitat Rovira i Virgili, 43007 Tarragona, Spain
@ email: m.kamal.sarker@gmail.com
@ Date: 23.05.2017
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

############### plot function for  confusion_matrix  ##################################
def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=10) #,fontweight='bold'
    plt.yticks(tick_marks, classes, rotation=90,fontsize=10)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm= np.around(cm,decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", fontsize=10, #fontweight='bold',
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=10) #,fontweight='bold'
    plt.xlabel('Predicted label', fontsize=10)

###########################################################################################