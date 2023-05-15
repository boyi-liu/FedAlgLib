import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parameters_to_tensor(self):
        """
            collect each param, then flatten them all

        :return: a list of flatten param
        """
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        params = torch.cat(params, 0)
        return params

    def tensor_to_parameters(self, tensor):
        """
            put a list of flatten param into a model

        :param tensor: the tensor to stored in parameters
        :return: None
        """
        param_index = 0
        for param in self.parameters():
            # === get shape & total size ===
            shape = param.shape
            param_size = 1
            for s in shape:
                param_size *= s

            # === put value into param ===
            param.data = tensor[param_index: param_index+param_size].view(shape).detach().clone()
            param_index += param_size


class CNNCifar(BaseModule):
    def __init__(self, args, dim_out):
        super(CNNCifar, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, dim_out)
        self.cls = dim_out

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]
        self.w_glob_keys = []
        if self.args.dataset == 'cifar10':
            for i in [0, 1, 3, 4]:
                self.w_glob_keys.extend(self.weight_keys[i])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class MLP(BaseModule):
    def __init__(self, args, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.args = args
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 64)
        self.layer_out = nn.Linear(64, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']] # Linear has weight and bias
        self.w_glob_keys = []
        if self.args.dataset == 'mnist':
            for i in [0, 1, 2]:
                self.w_glob_keys.extend(self.weight_keys[i])

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        x = self.relu(x)
        x = self.layer_out(x)
        return self.softmax(x)