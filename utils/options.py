import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--total_num', type=int, default=10, help="total clients num")
    parser.add_argument('--selected_rate', type=float, default=0.6, help="selected clients rate")
    parser.add_argument('--suffix', type=str, default='')

    # ===== Hardware Setting =====
    parser.add_argument('--device', type=int, default=0, help="which device to use")
    parser.add_argument('--gpu', type=int, default=0, help="which gpu")

    # ===== Method Setting ======
    parser.add_argument('--alg', type=str, default='fedavg', help="algorithm")
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='mlp', help="model for training")

    # ===== Training Setting =====
    parser.add_argument('--comm_round', type=int, default=1, help="communication round num")
    parser.add_argument('--local_batch_size', type=int, default=10, help="local batch size")
    parser.add_argument('--local_epochs', type=int, default=3, help="local epoch num")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

    # ===== Non-IID Setting =====
    parser.add_argument('--partition', type=str, default='iid', help="how to partition the dataset")  # iid/label-dir/label-class
    parser.add_argument('--dir_alpha', type=float, default=0.1, help="alpha in dirichlet distribution")
    parser.add_argument('--class_per_client', type=int, default=6, help="class_per_client")

    # ===== Asynchronous Setting =====
    parser.add_argument('--lag_level', type=float, default=1, help="lag level of device")
    parser.add_argument('--lag_rate', type=float, default=0.3, help="proportion of lag device")
    parser.add_argument('--is_async', type=bool, default=False, help="async mode")
    parser.add_argument('--mix_coe', type=float, default=0.9, help="mixing coefficient of async fl")

    # ===== Specific Algorithm =====
    # === FedProx ===
    parser.add_argument('--mu', type=float, default=0.1, help="FedProx-mu")
    # === FedRep ===
    parser.add_argument('--head_epochs', type=int, default=10, help='FedRep-head epochs')
    # === Cluster-Based ===
    parser.add_argument('--cluster_num', type=int, default=3, help='cluster num')

    args = parser.parse_args()
    return args
