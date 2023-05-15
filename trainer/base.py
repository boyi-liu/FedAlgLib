import random
import torch
import torch.nn as nn

from src.utils.modelloader import load_model
from torch.utils.data import DataLoader

from src.utils.dataprocess import Processor


class BaseClient:
    def __init__(self, id, args, dataset):
        self.id = id
        self.args = args
        self.dataset_train = dataset['train']
        self.dataset_test = dataset['test']
        self.device = args.device

        self.lr = args.lr
        self.batch_size = args.local_batch_size
        self.epoch = args.local_epochs
        self.model = load_model(args=args).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.5)

        self.meters = {'acc': Processor(), 'loss': Processor()}

        if self.dataset_train is not None:
            self.loader_train = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=None
            )
        if self.dataset_test is not None:
            self.loader_test = DataLoader(
                dataset=self.dataset_test,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=None
            )

    def local_train(self):
        raise NotImplementedError()

    def clone_model(self, target):
        p_tensor = target.model.parameters_to_tensor()
        self.model.tensor_to_parameters(p_tensor)
        return

    def local_test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.loader_test:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100.00 * correct / total


class BaseServer(BaseClient):
    def __init__(self, id, args, dataset):
        super().__init__(id, args, dataset)
        self.lr = args.lr
        self.client_num = args.total_num
        self.sample_rate = args.sample_rate
        self.clients = []
        self.sampled_clients = []

    def aggregate(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def test_all(self):
        for client in self.clients:
            if client in self.sampled_clients:
                self.meters['acc'].append(client.meters['acc'].last())
                self.meters['loss'].append(client.meters['loss'].last())
            else:
                self.meters['acc'].append(client.local_test())
        return self.meters['acc'].avg(), self.meters['loss'].avg()

    def sample(self):
        sample_num = self.sample_rate * self.client_num
        self.sampled_clients = random.sample(self.clients, sample_num)
