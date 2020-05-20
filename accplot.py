
import matplotlib.pyplot as plt


def loss_accuracy_plot(epoch_num, train_loss, val_loss,
                          train_acc,
                          val_acc):
    epoch_list=list(range(epoch_num))
    plt.plot(epoch_list,train_loss)
    plt.show()
    plt.savefig("t_l.png")

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
