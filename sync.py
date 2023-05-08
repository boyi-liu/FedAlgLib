import importlib


class Sync:
    def __init__(self, args):
        trainer_module = importlib.import_module(f'src.trainer.{args.alg}')

        self.server = eval('trainer_module.Server')(0, args, )