import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import monitoring_util as mu
import random
import pickle
import numpy as np

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
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # x = F.log_softmax(x)
        return x

def get_representations(includedDigits, network):
    left_image_indexes = []
    left_image_class = []
    left_representations = []
    left_values = []
    representation_dict = {}
    for tag in includedDigits:
        representation_dict[tag] = ([],[]) # representation, image vector

    dataset = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
    for idx in range(len(dataset)):
        item = dataset[idx]
        if item[1] in includedDigits:
            representation = network.forward(item[0].unsqueeze(0))
            representation = representation.squeeze(0)
            representation = representation.detach().numpy()
            left_representations.append(representation)
            left_values.append(item[0])
            left_image_class.append(item[1])
            # representation_entry = (representation,item[0])
            # left_representations.append(representation_entry)
            # representation_dict[item[1]][0].append(representation)
            # representation_dict[item[1]][1].append(item[0])

            # left_representations.append(representation)
            # left_image_indexes.append(idx)
            # left_image_class.append(item[1])

    representation_object = (left_representations,left_values,left_image_class)
    return representation_object

def main():
    includedDigits = [0,1,2,3,4]
    model_path = 'results/model_left.pth'
    network = Net()
    network.load_state_dict(torch.load(model_path))
    network.eval()
    left_representations = get_representations(includedDigits, network)
    with open('results/representations.pkl','wb') as pickleFile:
        pickle.dump(left_representations,pickleFile)

main()