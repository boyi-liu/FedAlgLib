class BaseClient:
    def __init__(self, id, args, dataset):
        self.id = id
        self.epoch = args.local_epochs
        self.batch_size = args.local_batch_size
        self.device = args.device


class BaseServer: