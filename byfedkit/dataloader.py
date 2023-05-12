import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np

MNIST_PATH = 'data/mnist/'
CIFAR10_PATH = 'data/cifar10/'

MNIST = 'mnist'
CIFAR10 = 'cifar10'

trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])


def _dataset(dataset):
    """
    load training dataset and test dataset
    support:
        MNIST,
        CIFAR-10

    :param dataset: indicator of dataset
    :return: true dataset
    """
    ret_dict = dict()
    if dataset == MNIST:
        ret_dict['train'] = datasets.MNIST(MNIST_PATH, train=True, download=True, transform=trans_mnist)
        ret_dict['test'] = datasets.MNIST(MNIST_PATH, train=False, download=True, transform=trans_mnist)
        return ret_dict
    elif dataset == CIFAR10:
        ret_dict['train'] = datasets.CIFAR10(CIFAR10_PATH, train=True, download=True, transform=trans_cifar10_train)
        ret_dict['test'] = datasets.CIFAR10(CIFAR10_PATH, train=False, download=True, transform=trans_cifar10_val)
        return ret_dict
    else:
        print('wrong dataset')
        exit(-1)


def _index_dict(dataset):
    """
    get {index: [labels]} dict of a dataset
    the key is a certain label of the dataset, the value is the index list of this label in the dataset

    example:
        {1:[1,3,4], 2:[2,5,7],...}

    :param dataset: given dataset
    :return: index->label dict
    """
    index_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()

        if label not in index_dict.keys():
            index_dict[label] = []
        index_dict[label].append(i)
    return index_dict


def _index2shards(index_dict, shard_num):
    """
    switch the index dict to shards
    split each index array to multiple arrays, and join them into a whole one

    example:
        {1:[1,3,4,8], 2:[2,5,7,9]...} -> [[1,3],[4,8],[2,5],[7,9]]

    :param index_dict: index dict to switch
    :param shard_num: how many shard should one index array be divided
    :return: final list of shards
    """
    shards = []
    for k in sorted(index_dict.keys()):
        index_k = np.array(index_dict[k])
        np.random.shuffle(index_k)
        chunks = np.array_split(index_k, shard_num)
        for _ in chunks:
            shards.append(_)
    return shards


def _iid_index(client_num, dataset):
    """
    get training index and test index under IID setting, where all clients get index containing all labels

    :param client_num: client num
    :param dataset: dataset dict, consist of train & test
    :return: training index & test index
    """
    dataset_train = dataset['train']
    dataset_test = dataset['test']

    shard_num = client_num

    index_dict_train = _index_dict(dataset_train)
    index_dict_test = _index_dict(dataset_test)

    shards_train = _index2shards(index_dict=index_dict_train, shard_num=shard_num)
    shards_test = _index2shards(index_dict=index_dict_test, shard_num=shard_num)

    class_num = len(index_dict_train.keys())

    select_list = np.arange(client_num * class_num)

    finals_train = [[] for _ in range(client_num)]
    finals_test = [[] for _ in range(client_num)]

    for ind, s in enumerate(select_list):
        client_id = ind % client_num
        finals_train[client_id].extend(shards_train[s])
        finals_test[client_id].extend(shards_test[s])

    return finals_train, finals_test


def _dirichlet_index(client_num, dataset, alpha):
    """
    get training index and test index under Dirichlet setting, following Dir(alpha)

    :param client_num:
    :param dataset: dataset dict, consist of train & test
    :param alpha: concentration variable in Dirichlet setting
    :return: training index & test index
    """
    dataset_train = dataset['train']
    dataset_test = dataset['test']

    index_dict_train = _index_dict(dataset_train)
    index_dict_test = _index_dict(dataset_test)

    class_num = len(index_dict_train.keys())

    dirichlet_pdf = np.random.dirichlet([alpha / class_num] * class_num, client_num)

    shard_num_train = len(dataset_train) // client_num
    shard_num_test = len(dataset_test) // client_num

    # === training dataset ===
    index_train_final = []
    for i in np.arange(client_num):
        _index_train = []
        local_dirichlet_pdf = dirichlet_pdf[i]
        local_pdf = np.floor(local_dirichlet_pdf * shard_num_train)
        for k in sorted(index_dict_train.keys()):
            index_k = np.array(index_dict_train[k])
            np.random.shuffle(index_k)
            _index_train.extend(index_k[:int(local_pdf[k])])
        index_train_final.append(_index_train)
    # === test dataset ===
    index_test_final = []
    for i in np.arange(client_num):
        _index_test = []
        local_dirichlet_pdf = dirichlet_pdf[i]
        local_pdf = np.floor(local_dirichlet_pdf * shard_num_test)
        for k in sorted(index_dict_test.keys()):
            index_k = np.array(index_dict_test[k])
            np.random.shuffle(index_k)
            _index_test.extend(index_k[:int(local_pdf[k])])
        index_test_final.append(_index_test)
    return index_train_final, index_test_final


def _imbalance_class_index(client_num, dataset, class_per_client):
    dataset_train = dataset['train']
    dataset_test = dataset['test']

    index_dict_train = _index_dict(dataset_train)
    index_dict_test = _index_dict(dataset_test)

    class_num = len(index_dict_train.keys())
    shard_num = class_per_client * client_num // class_num

    shards_train = _index2shards(index_dict=index_dict_train, shard_num=shard_num)
    shards_test = _index2shards(index_dict=index_dict_test, shard_num=shard_num)

    select_list = np.arange(client_num*class_per_client)
    np.random.shuffle(select_list)

    finals_train = [[] for _ in range(client_num)]
    finals_test = [[] for _ in range(client_num)]

    for ind, s in enumerate(select_list):
        client_id = ind % client_num
        finals_train[client_id].extend(shards_train[s])
        finals_test[client_id].extend(shards_test[s])

    return finals_train, finals_test


def load_mnist_index_iid(client_num):
    dataset = _dataset(MNIST)
    return _iid_index(client_num=client_num,
                      dataset=dataset)


def load_mnist_iid(client_num):
    dataset = _dataset(MNIST)
    index_train, index_test = _iid_index(client_num=client_num,
                                         dataset=dataset)
    return dataset['train'], dataset['test'], index_train, index_test


def load_cifar_index_iid(client_num):
    dataset = _dataset(CIFAR10)
    return _iid_index(client_num=client_num,
                      dataset=dataset)


def load_cifar10_iid(client_num):
    dataset = _dataset(CIFAR10)
    index_train, index_test = _iid_index(client_num=client_num,
                                         dataset=dataset)
    return dataset['train'], dataset['test'], index_train, index_test


def load_cifar10_dirichlet(client_num, alpha=0.1):
    dataset = _dataset(CIFAR10)
    index_train, index_test = _dirichlet_index(client_num=client_num,
                                               dataset=dataset,
                                               alpha=alpha)
    return dataset['train'], dataset['test'], index_train, index_test


def load_cifar10_class_imbalance(client_num, class_per_client):
    dataset = _dataset(CIFAR10)
    index_train, index_test = _imbalance_class_index(client_num=client_num,
                                                     dataset=dataset,
                                                     class_per_client=class_per_client)
    return dataset['train'], dataset['test'], index_train, index_test


def load_mnist_index(client_num, class_per_client, class_num):
    """
    load random index of mnist, not the actual dataset
    :param client_num:
    :param class_per_client:
    :param class_num:
    :return:
    """
    dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)

    shard_num = class_per_client * client_num // class_num

    index_dict_train = _index_dict(dataset_train)
    index_dict_test = _index_dict(dataset_test)

    shards_train = []
    shards_test = []

    for k in sorted(index_dict_train.keys()):
        index_k = np.array(index_dict_train[k])
        np.random.shuffle(index_k)

        chunks = np.array_split(index_k, shard_num)
        for _ in chunks:
            shards_train.append(_)

    for k in sorted(index_dict_test.keys()):
        index_k = np.array(index_dict_test[k])
        np.random.shuffle(index_k)

        chunks = np.array_split(index_k, shard_num)
        for _ in chunks:
            shards_test.append(_)

    labels = np.arange(class_num)
    select_list = []
    for c_id in np.arange(client_num):
        select_list.extend(labels*class_num+c_id)
    print(select_list)
    select_list = np.arange(client_num*class_num)
    # for label in labels:
    #     select_list.append()
    # print(select_list)

    # np.random.shuffle(labels)
    # select_list = np.repeat(labels, shard_num)  # [1,1,3,3,2,2,4,4,6,6,8,8,...]
    # for ind, _ in enumerate(select_list):
    #     select_list[ind] = (_-1) * shard_num + ind % shard_num  # [0,1,4,5,2,3,...]

    finals_train = [[] for _ in range(client_num)]
    finals_test = [[] for _ in range(client_num)]

    for ind, s in enumerate(select_list):
        client_id = ind % client_num
        finals_train[client_id].extend(shards_train[s])
        finals_test[client_id].extend(shards_test[s])

    return finals_train, finals_test


if __name__ == '__main__':
    _imbalance_class_index(10, )