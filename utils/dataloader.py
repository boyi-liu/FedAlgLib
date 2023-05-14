import src.byfedkit.dataloader as loader


def load_data(args):
    if args.partition == 'iid':
        return load_data_iid(args)
    elif args.partition == 'class':
        return load_data_label_class(args)
    elif args.partition == 'dir':
        return load_data_label_dirichlet(args)
    else:
        exit('Error: unrecognized data partition method')


def load_data_iid(args):
    client_num = args.total_num
    if args.dataset == 'mnist':
        return loader.load_mnist_iid(client_num)
    elif args.dataset == 'cifar10':
        return loader.load_cifar10_iid(client_num)
    else:
        exit('Error: unrecognized dataset')


def load_data_label_class(args):
    client_num = args.total_num
    class_per_client = args.class_per_client
    if args.dataset == 'cifar10':
        return loader.load_cifar10_class_imbalance(client_num=client_num, class_per_client=class_per_client)
    else:
        exit('Error: unrecognized dataset')


def load_data_label_dirichlet(args):
    client_num = args.total_num
    alpha = args.dir_alpha
    if args.dataset == 'cifar10':
        return loader.load_cifar10_dirichlet(client_num=client_num, alpha=alpha)
    else:
        exit('Error: unrecognized dataset')