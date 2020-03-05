import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import monitoring_util as mu
import random

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

def main():
    batch_size_train = 64
    fullyIncludedDigits = [5, 6, 7, 8, 9]
    pinnedDigits = [0, 1, 2, 3, 4]
    pinnedCount = 30
    train_loader_left = load_Right_TrainData_avg(batch_size_train, fullyIncludedDigits,pinnedDigits,pinnedCount)
    print('pause')

main()