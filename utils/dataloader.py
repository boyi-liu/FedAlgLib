from src.byfedkit.dataloader import *


def load_data(args):
    if args.partition == 'iid':
        return load_data_iid(args)
    else:
        exit('Error: unrecognized data partition method')


def load_data_iid(args):
    client_num = args.total_num
    if args.dataset == 'mnist':
        return load_mnist_iid(client_num)
    elif args.dataset == 'cifar10':
        return load_cifar10_iid(client_num)
    else:
        exit('Error: unrecognized dataset')
