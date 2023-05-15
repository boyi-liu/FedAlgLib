import importlib
import sys
import time
import numpy as np

from utils.options import args_parser
from utils.dataloader import load_data
from utils.dataprocess import Processor
from tqdm import tqdm


class Sync:
    def __init__(self, args):
        # === save config ===
        self.args = args


        # === load trainer ===
        trainer_module = importlib.import_module(f'src.trainer.{args.alg}')

        # === load dataset ===
        dataset, data_index = load_data(args)

        # === init clients ===
        self.clients = []
        for idx in range(args.total_num):
            client = eval('trainer_module.Client')(idx, args, dataset)
            self.clients.append(client)

        # === init server ===
        self.server = eval('trainer_module.Server')(0, args, dataset)
        self.server.clients = self.clients

    def train(self):
        OUTPUT_CLIENT = False
        global_pro = Processor()
        output = sys.stdout
        try:
            for rnd in tqdm(self.args.comm_round, desc='Communication Round', leave=False):
                output.write(f'==========Round {rnd} begin==========\n')
                time_start = time.time()

                # === train ===
                self.server.train()
                time_end = time.time()

                # === output client info ===
                if OUTPUT_CLIENT:
                    for client in sorted(self.server.sampled_clients, key=lambda x: x.id):
                        client_info = [f'client-{client.id}']
                        for k, v in client.meters.items():
                            client_info.append('%s: %.5f' % (k, v.last()))
                        output.write(', '.join(client_info) + '\n')

                # === output server info ===
                acc_all, loss_all = self.server.test_all()
                global_pro.append(acc_all)
                output.write('server, accuracy: %.5f\n' % acc_all)
                output.write('total time: %.0f seconds\n' % (time_end - time_start))
                output.write(f'==========Round {rnd} end==========\n')
                output.flush()
        except KeyboardInterrupt:
            ...
        finally:
            acc_list = global_pro.data
            avg_count = 5
            acc_avg = np.mean(acc_list[-avg_count:])
            acc_std = np.std(acc_list[-avg_count:])
            acc_max = np.max(acc_list)
            output.write('==========Summary==========\n')
            for client in self.clients:
                client.clone_model(self.server)
                output.write('client %d, accuracy: %.5f\n' % (client.id, client.test_accuracy()))
            output.write('server, max accuracy: %.5f\n' % acc_max)
            output.write('server, final accuracy: %.5f +- %.5f\n' % (acc_avg, acc_std))
            output.write('===========================\n')


if __name__ == '__main__':
    sync = Sync(args=args_parser())