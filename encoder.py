from comet_ml import Experiment
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
import random
import copy

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

class EncoderNet(nn.Module):

    def __init__(self):
        super(EncoderNet, self).__init__()

        # observation_width = 10
        # connect_1_2 = 300
        # connect_2_3 = 300
        # connect_3_4 = 300
        # connect_4_5 = 784
        #
        # self.fc1 = nn.Linear(in_features=observation_width, out_features=connect_1_2)
        # self.relu1 = nn.ReLU(inplace=True)
        #
        # self.fc2 = nn.Linear(in_features=connect_1_2, out_features=connect_2_3)
        # self.relu2 = nn.ReLU(inplace=True)
        #
        # self.fc3 = nn.Linear(in_features=connect_2_3, out_features=connect_3_4)
        # self.relu3 = nn.ReLU(inplace=True)
        #
        # self.fc4 = nn.Linear(in_features=connect_3_4, out_features=connect_4_5)
        #
        # torch.nn.init.uniform_(self.fc1.weight, -1 * 0.5, 0.5)
        # torch.nn.init.uniform_(self.fc2.weight, -1 * 0.5, 0.5)
        # torch.nn.init.uniform_(self.fc3.weight, -1 * 0.5, 0.5)
        # torch.nn.init.uniform_(self.fc4.weight, -1 * 0.5, 0.5)

        observation_width = 10
        connect_1_2 = 100
        connect_2_3 = 100
        connect_3_4 = 784

        self.fc1 = nn.Linear(in_features=observation_width, out_features=connect_1_2)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(in_features=connect_1_2, out_features=connect_2_3)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(in_features=connect_2_3, out_features=connect_3_4)
        self.relu3 = nn.ReLU(inplace=True)

        torch.nn.init.uniform_(self.fc1.weight, -1 * 0.5, 0.5)
        torch.nn.init.uniform_(self.fc2.weight, -1 * 0.5, 0.5)
        torch.nn.init.uniform_(self.fc3.weight, -1 * 0.5, 0.5)

    def forward(self, x):

        # out = x
        # out = self.fc1(out)
        # out = self.relu1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.fc3(out)
        # out = self.relu3(out)
        # out = self.fc4(out)

        out = x
        out = self.fc1(out)
        out = self.relu1(out)
        out = F.dropout(out,0.5)
        out = self.fc2(out)
        out = self.relu2(out)
        out = F.dropout(out, 0.5)
        out = self.fc3(out)

        return out

def constructSource(network,includedDigits):

    representations = []

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
            # representation = representation.squeeze(0)
            # representation = representation.detach().numpy()
            representations.append((representation,item[0],item[1])) #representation, source, tag

    return representations

def prepareEncoderSource(includedDigits,model_left):
    network = Net()
    network.load_state_dict(torch.load(model_left))
    network.eval()

    source = constructSource(network, includedDigits)

    with open('results/encoder_source.pkl', 'wb') as pickleFile:
        pickle.dump(source, pickleFile)

def makeBatches(source,batch_size):

    batches = []

    source_temp = copy.deepcopy(source)

    while len(source_temp) > batch_size:
        r = random.randint(0,len(source_temp)-batch_size)
        for idx in range(r,r+batch_size):
            batches.append(source_temp[idx])
        for idx in range(r + batch_size-1,r-1,-1):
            del source_temp[idx]

    return batches

def trainEncoder(encoder, optimizer, epoch,learning_rate, momentum, batch_size, experiment, encoder_source_path, encoder_model_path):

    encoder.train()
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    with open(encoder_source_path, 'rb') as pickleFile:
        source = pickle.load(pickleFile)

    trimmed_source = []
    for item in source:
        source_values = item[0]
        tag =  torch.tensor(item[1], dtype=float, requires_grad=True).squeeze(0).float()
        trimmed_source.append((source_values,tag))

    # batches = makeBatches(source,batch_size)

    train_loader = torch.utils.data.DataLoader(trimmed_source, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # data = []
        # target = []
        # batch_source = batches[idx]
        # for item in batch_source:
        #     data.append(item[0])
        #     target.append(item[1])
        # data = torch.cat(data)
        # target = torch.cat(target)
        target = target.view((batch_size, 784))
        # data = torch.tensor(data, dtype=float, requires_grad=True).float()
        # target = torch.tensor(target, dtype=float, requires_grad=True).squeeze(0).float()
        output = encoder(data)
        loss = criterion(output, target)
        loss_value = loss.item()
        # experiment.log_metric("loss_vf",loss_value)
        print(f'Epoch: {epoch}  Batch: {batch_idx}  Loss: {loss_value}')
        loss.backward()
        optimizer.step()

    # for idx in range(len(batches)):
    #     optimizer.zero_grad()
    #     data = []
    #     target = []
    #     batch_source = batches[idx]
    #     for item in batch_source:
    #         data.append(item[0])
    #         target.append(item[1])
    #     data = torch.cat(data)
    #     target = torch.cat(target)
    #     target = target.view((batch_size, 784))
    #     data = torch.tensor(data, dtype=float, requires_grad=True).float()
    #     target = torch.tensor(target, dtype=float, requires_grad=True).squeeze(0).float()
    #     output = encoder(data)
    #     loss = criterion(output, target)
    #     loss_value = loss.item()
    #     # experiment.log_metric("loss_vf",loss_value)
    #     print(f'Epoch: {epoch}  Batch: {idx}  Loss: {loss_value}')
    #     loss.backward()
    #     optimizer.step()

    # sourceLength = len(source)
    # sourceLastBatch = sourceLength - (sourceLength % batch_size)
    #
    # for idx in range(0,sourceLastBatch,batch_size):
    #     optimizer.zero_grad()
    #     data = []
    #     target = []
    #     batch_source = source[idx:idx+batch_size]
    #     for item in batch_source:
    #         data.append(item[0])
    #         target.append(item[1])
    #     data = torch.cat(data)
    #     target = torch.cat(target)
    #     target = target.view((batch_size,784))
    #     data = torch.tensor(data,dtype=float,requires_grad=True).float()
    #     target = torch.tensor(target,dtype=float,requires_grad=True).squeeze(0).float()
    #     output = encoder(data)
    #     loss = criterion(output,target)
    #     loss_value = loss.item()
    #     # experiment.log_metric("loss_vf",loss_value)
    #     print(f'Epoch/Batch {epoch}/ {idx} of {sourceLength}  Loss: {loss_value}')
    #     loss.backward()
    #     optimizer.step()

    torch.save(encoder.state_dict(), encoder_model_path)

def viewEncodedImages(encoder_model_path,encoder_source_path,experiment):

    encoder = EncoderNet()
    encoder.load_state_dict(torch.load(encoder_model_path))
    encoder.eval()

    with open(encoder_source_path,'rb') as pickleFile:
        encoder_source = pickle.load(pickleFile)

    generated_images = []

    for idx in range(6):
        output = encoder.forward(encoder_source[idx][0])
        image = output.view((28,28)).detach().numpy()
        orig_image = encoder_source[idx][1].view((28,28)).detach().numpy()
        generated_images.append((image,orig_image))

    fig = plt.figure()
    for idx in range(6):
      plt.subplot(2,3,idx+1)
      plt.tight_layout()
      plt.imshow(generated_images[idx][1], cmap='gray', interpolation='none')
      plt.title("Original Image")
      plt.xticks([])
      plt.yticks([])
    fig.show()

    fig = plt.figure()
    for idx in range(6):
      plt.subplot(2,3,idx+1)
      plt.tight_layout()
      plt.imshow(generated_images[idx][0], cmap='gray', interpolation='none')
      plt.title("Generated Image")
      plt.xticks([])
      plt.yticks([])
    fig.show()

    experiment.log_figure(figure=fig)

def main():
    model_left = 'results/model_left.pth'
    includedDigits = [0, 1, 2, 3, 4]

    encoder_model_path = 'results/encoder.pth'
    encoder_source_path = 'results/encoder_source.pkl'

    n_epochs = 50
    batch_size_train = 8
    learning_rate = 0.0015
    momentum = 0.9
    optim_type = 'adam'
    weight_decay = 0.0

    # prepareEncoderSource(includedDigits,model_left)

    experiment = Experiment(api_key="1x1ZQpvbtvDyO2s5DrlUyYpzv",
                            project_name="general2", workspace="verlyn-fischer")
    hyper_params = {'dropout':0.5, "learning_rate": learning_rate, 'momentum':momentum, 'epochs': n_epochs, "batch_size": batch_size_train, 'network_shape':'10/100/100/784','optimizer':optim_type,'weight_decay':weight_decay}
    experiment.log_parameters(hyper_params)

    encoder = EncoderNet()
    if optim_type == 'adam':
        optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(1, n_epochs + 1):
        trainEncoder(encoder, optimizer, epoch, learning_rate, momentum, batch_size_train, experiment, encoder_source_path, encoder_model_path)

    viewEncodedImages(encoder_model_path,encoder_source_path,experiment)

    print('done')

main()