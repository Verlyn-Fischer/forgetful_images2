import numpy as np
from sklearn.linear_model import LinearRegression
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import salient_images as si

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])

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
        # x = F.log_softmax(x)
        return x

def get_representations(includedDigits, network):

    representations = []

    dataset = torchvision.datasets.MNIST('mnist_data', train=False, download=True,
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
            representations.append(representation)

    return representations

def constructPairs(model_left,model_right):
    includedDigits = [0, 1, 2, 3, 4]

    network = Net()
    network.load_state_dict(torch.load(model_left))
    network.eval()
    left_representations = get_representations(includedDigits, network)

    network = Net()
    network.load_state_dict(torch.load(model_right))
    network.eval()
    right_representations = get_representations(includedDigits, network)

    return left_representations, right_representations

def main():
    model_left = 'results/model_left.pth'
    model_right = 'results/model_right.pth'
    left_representations, right_representations = constructPairs(model_left,model_right)
    reg = LinearRegression().fit(left_representations, right_representations)
    print(f'Score: {reg.score(left_representations, right_representations)}')
    print(f'Coef: {reg.coef_}')
    print(f'Intercept: {reg.intercept_}')

main()
