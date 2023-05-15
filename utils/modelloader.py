from src.model import *

class_num_dict = {'mnist': 10, 'cifar10': 10}


def load_model(args):
    model_arg = args.model
    dataset_arg = args.dataset

    if model_arg == 'mlp' and dataset_arg == 'mnist':
        return MLP(args=args, dim_in=784, dim_hidden=256, dim_out=class_num_dict[dataset_arg])
    elif model_arg == 'cnn' and dataset_arg == 'cifar10':
        return CNNCifar(args=args, dim_out=class_num_dict[dataset_arg])
    else:
        exit('Error: unrecognized model')