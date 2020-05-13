    import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    cm1 = cm
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100,0)

    else:
        print('Confusion matrix, without normalization')
    #classes=['ZERO','ONE','TWO','THREE']
    #print(y_true)
    #print(y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            num = format(cm[i, j], fmt)
            num = num.split(".")[0]
            ax.text(j, i, str(cm1[i, j]) + " (" + num+"%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize = 8)

    fig.tight_layout()
    return ax

'''
EXAMPLE CODE (BELOW)
'''
def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    np.set_printoptions(precision=2)
    print(y_test)
    print(y_pred)
    print(class_names)
    print(y_test.type())
    print(class_names.type())
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True)

    plt.savefig("confusion_matrix.png")
    return

if __name__ == "__main__":
    main()
