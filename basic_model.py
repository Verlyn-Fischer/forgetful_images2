import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import monitoring_util as mu
import random
import pickle

def plotGroundTruth(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    fig = plt.figure()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(example_targets[i]))
      plt.xticks([])
      plt.yticks([])
    fig.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train(epoch, log_interval, train_loader, experiment, network, optimizer):
    train_losses = []
    train_counter = []
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          train_counter_value = (batch_idx*64) + ((epoch-1)*len(train_loader.dataset))
          mu.writeLoss(loss.item(),train_counter_value,experiment)
          train_losses.append(loss.item())
          train_counter.append(train_counter_value)
          torch.save(network.state_dict(), 'results/model.pth')
          torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test(test_loader,network):
    test_losses = []
    correct_calls = [0,0,0,0,0,0,0,0,0,0]
    incorrect_calls = [0,0,0,0,0,0,0,0,0,0]
    # test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
          output = network(data)
          test_loss += F.nll_loss(output, target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          for data_index in range(len(target)):
              target_element = target[data_index]
              target_pred = pred[data_index]
              idx = target_element.item()
              predValue = target_pred.item()
              if predValue == idx:
                  correct_calls[idx] = correct_calls[idx] + 1
              else:
                  incorrect_calls[idx] = incorrect_calls[idx] + 1

          correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        mu.plotAccuracy(correct_calls,incorrect_calls)

def load_Left_TrainData(batch_size_train,includedDigits):
    left_set_indices = []
    dataset = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item[1] in includedDigits:
            left_set_indices.append(idx)

    subsetSampler_left = torch.utils.data.SubsetRandomSampler(left_set_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=False,
                                               sampler=subsetSampler_left)
    return train_loader

def load_Right_TrainData(batch_size_train,fullyIncludedDigits,pinnedDigits,pinnedRate):
    set_indices = []
    dataset = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item[1] in fullyIncludedDigits:
            set_indices.append(idx)
        elif item[1] in pinnedDigits:
            r = random.uniform(0,1)
            if r < pinnedRate:
                set_indices.append(idx)

    subsetSampler_right = torch.utils.data.SubsetRandomSampler(set_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=False,
                                               sampler=subsetSampler_right)
    return train_loader

def load_Right_TrainData_avg(batch_size_train, includedDigits, pinnedDigits, pinnedCount):

    average_images = [None,None,None,None,None,None,None,None,None]
    image_count = [0,0,0,0,0,0,0,0,0,0]

    left_set_indices = []
    dataset = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))

    averaged_dataset = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item[1] in includedDigits:
            averaged_dataset.append(item)
        elif item[1] in pinnedDigits:
            image_count[item[1]] += 1
            if average_images[item[1]] is None:
                average_images[item[1]] = item[0]
            else:
                average_images[item[1]] = average_images[item[1]] + item[0]

    for idx in range(len(average_images)):
        if average_images[idx] is not None:
            avg_image = average_images[idx] / image_count[idx]
            for iteration in range(pinnedCount):
                averaged_dataset.append((avg_image,idx))

    train_loader = torch.utils.data.DataLoader(averaged_dataset, batch_size=batch_size_train, shuffle=True)
    return train_loader

def load_Right_salientPins(batch_size_train, fullyIncludedDigits, pinnedCount, pinnedFile):

    dataset_new = []

    dataset = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))

    for idx in range(len(dataset)):
        item = dataset[idx]
        if item[1] in fullyIncludedDigits:
            dataset_new.append((item[0],item[1]))

    with open(pinnedFile, 'rb') as pickleFile:
        pinnedSet = pickle.load(pickleFile)

    for pin in pinnedSet:
        for count in range(pinnedCount):
            dataset_new.append(pin)

    train_loader = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size_train, shuffle=True)

    return train_loader

def load_Right_TrainData_OLD(batch_size_train,fullyIncludedDigits,pinnedDigits):
    ##### TRAIN ############

    weights = []
    dataset = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))

    for idx in range(len(dataset)):
        item = dataset[idx]
        if item[1] in fullyIncludedDigits:
            weights.append(1.0)
        elif item[1] in pinnedDigits:
            weights.append(0.1)
        else:
            weights.append(0.0)

    num_samples = 5000
    batch_size = 64
    subsetSampler_right = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=False)
    batchSample_right = torch.utils.data.BatchSampler(subsetSampler_right, batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batchSample_right)

    return train_loader

def loadTestData(batch_size_test):
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    return test_loader

def main():
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    experiment = 'Pin_cluster2'

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # LEFT CONTROLS
    # includedDigits_left = [0,1,2,3,4]
    # train_loader_left = load_Left_TrainData(batch_size_train, includedDigits_left)

    # RIGHT CONTROLS
    fullyIncludedDigits = [5,6,7,8,9]
    pinnedDigits = [0,1,2,3,4]
    pinnedRate = 0.01
    pinnedCount = 1
    pinFile = 'results/pin_set.pkl'
    # train_loader_right = load_Right_TrainData(batch_size_train, fullyIncludedDigits, pinnedDigits,pinnedRate)
    # train_loader_right = load_Right_TrainData_avg(batch_size_train, fullyIncludedDigits, pinnedDigits, pinnedCount)
    train_loader_right = load_Right_salientPins(batch_size_train, fullyIncludedDigits, pinnedCount, pinFile)

    test_loader = loadTestData(batch_size_test)

    # SEQUENCES

    # No Training
    # test(test_loader,network)

    # Left Training
    # for epoch in range(1, n_epochs + 1):
    #     train(epoch, log_interval, train_loader_left, experiment, network, optimizer)
    # test(test_loader, network)
    # torch.save(network.state_dict(), 'results/model_left.pth')

    # Right Training
    model_path = 'results/model_left.pth'
    network.load_state_dict(torch.load(model_path))
    test(test_loader, network)
    for epoch in range(1, n_epochs + 1):
        train(epoch, log_interval, train_loader_right, experiment, network, optimizer)
    test(test_loader, network)
    torch.save(network.state_dict(), 'results/model_right.pth')

main()