import random


class BaseClient:
    def __init__(self, id, args, dataset):
        self.id = id
        self.args = args
        self.dataset
        self.epoch = args.local_epochs
        self.batch_size = args.local_batch_size
        self.device = args.device
        self.clients = []


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
        return random.sample(self.clients)