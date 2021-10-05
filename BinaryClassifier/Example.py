import collections

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
import numpy as np


# Code Description:
# U can choose nn.Softmax()[also change threshold from 0 to number greater then 0]
# or torch.norm to using different 0&1 transaction strategy

class Net(torch.nn.Module):
    def __init__(self, n_feature=2, n_hidden=8, n_output=2):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

    def train(self, train_x, train_y, epoch=1000):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.02)
        loss_func = torch.nn.CrossEntropyLoss()
        for t in range(epoch):
            out = self(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def save(self, path):
        torch.save(self, path)

class MaskGenerator(torch.nn.Module):
    def __init__(self, n_nodes, embedding_dim, in_channels = 2, out_dim = 2, hidden_dim = 8, n_hidden=1,
                 spec_norm = False):
        super(MaskGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )
        # print(len(layers))
        self.mlp = nn.Sequential(*layers)
        self.l1_weights = nn.Linear(hidden_dim, hidden_dim * self.in_channels)
        self.l1_bias = nn.Linear(hidden_dim, hidden_dim)
        self.l2_weights = nn.Linear(hidden_dim, hidden_dim * self.out_dim)
        self.l2_bias = nn.Linear(hidden_dim, out_dim)
        if spec_norm:
            self.l1_weights = spectral_norm(self.l1_weights)
            self.l1_bias = spectral_norm(self.l1_bias)
            self.l2_weights = spectral_norm(self.l2_weights)
            self.l2_bias = spectral_norm(self.l2_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)
        # print(self.l1_weights(features).shape)
        # print(self.l2_weights(features).shape)
        weights = collections.OrderedDict({
            "hidden.weight": self.l1_weights(features).view(self.hidden_dim, self.in_channels),
            "hidden.bias": self.l1_bias(features).view(-1),
            "out.weight": self.l2_weights(features).view(self.out_dim, self.hidden_dim),
            "out.bias": self.l2_bias(features).view(-1),
        })
        return weights

    def train(self, idx: int, x, y, path, epoch=100):
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
        stat_list = []
        masked_acc_list = []
        static_acc_list = []
        for t in range(epoch):
            # Our works:

            # load meta model
            meta = torch.load(path)
            meta_optimizer = torch.optim.Adam(meta.parameters(), lr=0.02)
            # print('Unmasked Loss: {}'.format(loss_func(meta(x), y)))
            stat_list.append(loss_func(meta(x), y).detach().numpy())
            static_acc_list.append(getAccuracy(meta, x, y))
            unmasked_weights = meta.state_dict()
            # generate mask
            pred = self(idx)
            # print(pred)
            mask = collections.OrderedDict()
            # print(mask)
            for n, v in pred.items():
                mask[n] = self.isZero(self.softmax_norm(pred[n]))
            # print(mask)
            # masked weights
            masked_weights = collections.OrderedDict()
            # mask behavior:
            # print('[INFO] Masked State Before: {}'.format(meta.state_dict()))
            # print('-- Round {}'.format(t))
            for k, _ in unmasked_weights.items():
                masked_weights[k] = mask[k] * unmasked_weights[k]
                # print('-- {} layers: {} * {} = {}'.format(k, mask[k].shape, unmasked_weights[k].shape, masked_weights[k].shape))
            # load updated parameters
            meta.load_state_dict(masked_weights)
            # print('[INFO] Masked State After: {}'.format(meta.state_dict()))
            out = meta(x)
            loss = loss_func(out, y)
            loss_list.append(loss.detach().numpy())
            masked_acc_list.append(getAccuracy(meta, x, y))
            meta_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta.parameters(), 50)
            meta_optimizer.step()
            # print('Masked Loss {}'.format(loss))
            # meta model updated
            optimizer.zero_grad()
            final_state = meta.state_dict()
            delta_theta = collections.OrderedDict({k: masked_weights[k] - final_state[k] for k in mask.keys()})
            # print(delta_theta)
            # calculating phi gradient
            # for v in mask.values():
            #    print('--> Mask {}'.format(v))
            # for param in self.parameters():
            #    print('<-- Self {} {}'.format(type(param), param.size()))
            # print(delta_theta)

            # bugs here
            # print()
            hnet_grads = torch.autograd.grad(
                list(pred.values()), self.parameters(), grad_outputs=list(delta_theta.values())
            )
            # print(hnet_grads)
            # update hnet weights
            for p, g in zip(self.parameters(), hnet_grads):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
            optimizer.step()
            # print()
        # print(loss_list)
        # print(len([i for i in range(epoch)]))
        plt.figure(1)
        plt.plot([i for i in range(epoch)], masked_acc_list, 'g--', label='masked model')
        plt.plot([i for i in range(epoch)], static_acc_list, 'r--', label='meta model')
        plt.legend(['Mask Acc', 'Meta Acc'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('accuracy(%)')

        plt.figure(2)
        plt.plot([i for i in range(epoch)], loss_list, 'g--', label='masked model')
        plt.plot([i for i in range(epoch)], stat_list, 'r--', label='meta model')
        plt.legend(['Mask Loss', 'Meta Loss'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('LOSS')
        plt.show()
    def softmax_norm(self, input):
        dim = input.shape
        size = len(dim)
        zero = nn.Softmax(dim=0)
        one = nn.Softmax(dim=1)
        if size == 1:
            input = zero(input)
        else :
            input = one(input)
        return input
    def isZero(self, input):
        threshold = 0.2
        dim = input.shape
        size = len(dim)
        if size == 1:
            for i in range(len(input)):
                input[i] = 1 if input[i] > threshold else 0
        else:
            for i in range(dim[0]):
                for j in range(dim[1]):
                    input[i][j] = 1 if input[i][j] > threshold else 0
        return input
def getAccuracy(model: nn.Module, x, y):
    result_list = [0, 1]
    list = []
    pred = model(x)
    for v in pred:
        if v[0] > v[1]:
            list.append(0)
        else:
            list.append(1)
    np_list = np.array(list)
    label_list = y.detach().numpy()
    pred_list = y.detach().numpy()
    acc = 0
    for i in range(len(pred_list)):
        if label_list[i] == pred_list[i]:
            acc += 1
    # print('{}/{} = {}%'.format(acc, len(pred_list), acc * 100 / len(pred_list)))
    return acc * 100 / len(pred_list)

n_data = torch.ones(100, 2)
x0 = torch.normal(3*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-3*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# omit
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# ML training done
net = Net()
net.train(x,y)
net.save('temp.pt')
unmasked_state = net.state_dict()
# print(unmasked_state)

acc = getAccuracy(net, x, y)

mask = collections.OrderedDict()
for key, value in unmasked_state.items():
    if len(value.shape) == 1:
        mask[key] = torch.ones(len(value))
    else :
        mask[key] = torch.ones(len(value), len(value[0]))
# print(mask)

node = 1
embed_dim = int ( 1 + node / 4)
hnet = MaskGenerator(node, embed_dim)
hnet.train(torch.tensor([0]), x, y, 'temp.pt')