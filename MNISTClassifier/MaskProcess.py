import collections

import torch
from torch import optim
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class HyperNet(nn.Module):
    def __init__(self, n_nodes, embedding_dim, in_channels = 1, n_kernels = 10, out_dim = 10, hidden_dim = 100, n_hidden=1,
                 spec_norm = False):
        super(HyperNet, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )
        self.mlp = nn.Sequential(*layers)

        self.conv1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
        self.conv1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.conv2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
        self.conv2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.fc1_weights = nn.Linear(hidden_dim, 320 * 50)
        self.fc1_bias = nn.Linear(hidden_dim, 50)
        self.fc2_weights = nn.Linear(hidden_dim, 10 * 50)
        self.fc2_bias = nn.Linear(hidden_dim, 10)

        if spec_norm:
            self.conv1_weights = spectral_norm(self.conv1_weights)
            self.conv1_bias = spectral_norm(self.conv1_bias)
            self.conv2_weights = spectral_norm(self.conv2_weights)
            self.conv2_bias = spectral_norm(self.conv2_bias)
            self.fc1_weights = spectral_norm(self.fc1_weights)
            self.fc1_bias = spectral_norm(self.fc1_bias)
            self.fc2_weights = spectral_norm(self.fc2_weights)
            self.fc2_bias = spectral_norm(self.fc2_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = collections.OrderedDict({
            "conv1.weight": self.conv1_weights(features).view(self.n_kernels, self.in_channels, 5, 5),
            "conv1.bias": self.conv1_bias(features).view(-1),
            "conv2.weight": self.conv2_weights(features).view(2 * self.n_kernels, self.n_kernels, 5, 5),
            "conv2.bias": self.conv2_bias(features).view(-1),
            "fc1.weight": self.fc1_weights(features).view(50, 320),
            "fc1.bias": self.fc1_bias(features).view(-1),
            "fc2.weight": self.fc2_weights(features).view(10, 50),
            "fc2.bias": self.fc2_bias(features).view(-1),
        })
        return weights

    def train(self, idx: int, train_loader, path, epoch=10):
        optimizers = {
            'sgd': torch.optim.SGD(
                [
                    {'params': [p for n, p in self.named_parameters() if 'embed' not in n]},
                    {'params': [p for n, p in self.named_parameters() if 'embed' in n], 'lr': 0}
                ], lr=1e-2, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=self.parameters(), lr=1e-2)
        }
        optimizer = optimizers['adam']
        loss_func = torch.nn.CrossEntropyLoss()

        loss_list = []

        temp = torch.load(path)
        acc = getAccuracy(temp, batch_size_train, train_loader)
        static_acc_list = [acc for i in range(epoch)]

        stat_list = []
        masked_acc_list = []
        for t in tqdm(range(epoch)):
            round_loss = 0
            unmasked_loss = 0
            acc = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                # Our works:

                # load meta model
                meta = torch.load(path)
                meta_optimizer = torch.optim.Adam(meta.parameters(), lr=0.02)
                unmasked_loss = loss_func(meta(data), target)
                unmasked_weights = meta.state_dict()
                # generate mask
                pred = self(idx)
                # print(pred)
                mask = collections.OrderedDict()
                for n, v in pred.items():
                    mask[n] = self.isZero(self.softmax_norm(pred[n]))
                # masked weights
                masked_weights = collections.OrderedDict()
                # mask behavior:
                for k, _ in unmasked_weights.items():
                    masked_weights[k] = mask[k] * unmasked_weights[k]
                # load updated parameters
                meta.load_state_dict(masked_weights)
                out = meta(data)
                acc += getBatchAccuracy(out, target,len(target))
                loss = loss_func(out, target)
                round_loss += loss
                meta_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(meta.parameters(), 50)
                meta_optimizer.step()
                # meta model updated
                optimizer.zero_grad()
                final_state = meta.state_dict()
                delta_theta = collections.OrderedDict({k: masked_weights[k] - final_state[k] for k in mask.keys()})
                hnet_grads = torch.autograd.grad(
                    list(pred.values()), self.parameters(), grad_outputs=list(delta_theta.values())
                )
                # update hnet weights
                for p, g in zip(self.parameters(), hnet_grads):
                    p.grad = g
                torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
                optimizer.step()
            round_loss /= len(train_loader)
            loss_list.append(round_loss.detach().numpy())
            stat_list.append(unmasked_loss.detach().numpy())
            acc /= 600
            masked_acc_list.append(acc)
            print('Round {} average loss: {}'.format(t, round_loss))
        plt.figure(1)
        plt.plot([i for i in range(epoch)], loss_list, 'g--', label='masked model')
        plt.plot([i for i in range(epoch)], stat_list, 'r--', label='meta model')
        plt.legend(['Mask Loss', 'Meta Loss'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        plt.figure(2)
        plt.plot([i for i in range(epoch)], masked_acc_list, 'g--', label='masked model')
        plt.plot([i for i in range(epoch)], static_acc_list, 'r--', label='meta model')
        plt.legend(['Mask Acc', 'Meta Acc'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('accuracy(%)')
        plt.show()

    def softmax_norm(self, input):
        dim = input.shape
        size = len(dim)
        m = nn.Softmax(dim = size - 1)
        input = m(input)
        return input

    def isZero(self, input):
        threshold = 0.5
        zero = torch.zeros_like(input)
        one = torch.ones_like(input)
        input = torch.where(input > threshold, one, input)
        input = torch.where(input <= threshold, zero, input)
        # print(input)
        return input

# init model
class CNN(nn.Module):
    def __init__(self, in_channel = 1, n_kernel = 10,kernel_size = 5, out_dim = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, n_kernel, kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def getAccuracy(model: nn.Module, batch_size, loader: DataLoader):
    model.eval()
    result_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    acc = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            mapping = torch.argmax(output, dim = 1)
            judge = mapping == target
            for v in judge:
                if v:
                    acc += 1
    # print('{}/{} = {}%'.format(acc, len(loader), acc * 100 / 60000))
    return acc * 100 / 60000

def getBatchAccuracy(out, target, batch_size):
    mapping = torch.argmax(out, dim=1)
    acc = 0
    judge = mapping == target
    for v in judge:
        if v:
            acc += 1
    return acc

batch_size_train = 64
batch_size_test = 1000
# load dataset:
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../mnist', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

net = torch.load('model.pt')
print(net)
# getAccuracy(net, batch_size_train, train_loader)


state_dict = net.state_dict()
for key, value in state_dict.items():
    print('{} --> {}'.format(key, value.shape))
hnet = HyperNet(n_nodes = 1, embedding_dim = 1)
print(hnet)

hnet.train(torch.tensor([0]), train_loader, 'model.pt')
