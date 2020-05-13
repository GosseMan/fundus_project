
import matplotlib.pyplot as plt


def loss_accuracy_plot(epoch_num, train_loss, val_loss,
                          train_acc,
                          val_acc):
    epoch_list=list(range(epoch_num))
    ax1=plt.subplot(2,1,1)
    plt.title('loss')
    graph1=plt.plot(epoch_list,train_loss,epoch_list,val_loss)
    ax2=plt.subplot(2,1,2)
    plt.title('accuracy')
    graph2=  plt.plot(epoch_list,train_acc,epoch_list,val_acc)
    plt.savefig("hhh.png")
    plt.show()
    return
def main():
    a=list(range(0,10))
    b=list(range(0,10))
    c=list(range(0,10))
    d=list(range(0,10))
    e=list(range(0,10))
    loss_accuracy_plot(10,b,c,d,e)
    plt.savefig("aaa.png")
    return

if __name__ == "__main__":
    main()
