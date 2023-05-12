import importlib
import sys
import time

from utils.options import args_parser
from utils.dataloader import load_data
from tqdm import tqdm


class Sync:
    def __init__(self, args):
        # === save config ===
        self.args = args

        # === load trainer ===
        trainer_module = importlib.import_module(f'src.trainer.{args.alg}')

        # === load dataset ===
        dataset_train, dataset_test, index_train, index_test = load_data(args)

        # === init clients ===
        self.clients = []
        for idx in range(args.total_num):
            client = eval('trainer_module.Client')(idx, args, dataset_train[idx], dataset_test[idx])
            self.clients.append(client)

        # === init server ===
        self.server = eval('trainer_module.Server')(0, args, )
        self.server.clients = self.clients

    def train(self):
        output = sys.stdout
        try:
            for rnd in tqdm(self.args.comm_round, desc='Communication Round', leave=False):
                output.write('==========Round %d begin==========\n' % rnd)
                time_start = time.time()

                # === train ===
                clients = self.server.train()


if __name__ == '__main__':
    sync = Sync(args=args_parser())