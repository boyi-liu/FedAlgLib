import random
import torch
from torch.utils.data import DataLoader

class BaseClient:
    def __init__(self, id, args, dataset):
        self.id = id
        self.args = args
        self.dataset_train = dataset['train']
        self.dataset_test = dataset['test']
        self.epoch = args.local_epochs
        self.batch_size = args.local_batch_size
        self.device = args.device
        self.clients = []

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



class BaseServer(BaseClient):
    def __init__(self, id, args, dataset):
        super().__init__(id, args, dataset)
        self.lr = args.lr

    def aggregate(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def sample(self):
        sample_num = self.args.selected_rate * self.args.total_num
        return random.sample(self.clients, sample_num)