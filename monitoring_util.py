from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

def writeLoss(loss,iteration,experiment):
    writer = SummaryWriter('logging/'+experiment)
    writer.add_scalar(tag='_losses',scalar_value=loss,global_step=iteration)
    writer.close()

def writeAccuracy(accuracy,iteration,experiment):
    writer = SummaryWriter('logging/'+experiment)
    writer.add_scalar(tag='_accuracies',scalar_value=accuracy,global_step=iteration)
    writer.close()

def plotAccuracy(correct_calls,incorrect_calls):
    ind = np.arange(10)
    plt.bar(ind, correct_calls, width=0.4, color='g')
    plt.bar(ind, incorrect_calls, bottom=correct_calls, width=0.4, color='r')
    plt.legend(['Correct','Incorrect'])
    plt.title('Predictions')
    plt.xlabel('Digit')
    plt.show()