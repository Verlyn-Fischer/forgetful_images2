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

def plotHist(annotated_reps):
    sequence = [[],[],[],[],[]]
    for item in annotated_reps:
        sequence[item[0]].append(item[1])
    # (tag,distance,source)
    plt.hist(sequence,bins=60,histtype='step')
    plt.legend(['0','1','2','3','4'])
    plt.show()