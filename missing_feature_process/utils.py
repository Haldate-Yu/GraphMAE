"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
Modified by Daeho Um (daehoum1@snu.ac.kr)
"""
import os
import csv
import torch
from torch_scatter import scatter_add


def get_missing_feature_mask(rate, n_nodes, n_features, seed, type="uniform"):
    """ 
    Return mask of shape [n_nodes, n_features] indicating whether each feature is present or missing.

    If `type`='uniform', then each feature of each node is missing uniformly at random with probability `rate`.
    Instead;

    if `type`='structural', either we observe all features for a node, or we observe none. For each node
    there is a probability of `rate` of not observing any feature.

    features with 'True' mask keep their original values,
    while features with 'False' mask will be changed according to fill method.
    """
    torch.manual_seed(seed)
    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    else:
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD


def save_node_results(args, mae_args, test_acc_mean, test_acc_std):
    result_dir = './results/'
    if not os.path.exists(result_dir):
        print("Creating directory for results ...")
        os.makedirs(result_dir)

    node_result_dir = os.path.join(result_dir, 'node_results/')
    if not os.path.exists(node_result_dir):
        print("Creating directory for node results ...")
        os.makedirs(node_result_dir)

    # parsing args
    result_file_name = (node_result_dir +
                        args.dataset_name + '_' +
                        args.filling_method + '_' +
                        args.mask_type + '_' +
                        str(args.missing_rate) + '.csv')

    if args.filling_method == 'graphmae':
        header_list = ["filling_method", "downstream_model", "n_runs", "num_layers", "hidden_dim", "jk",
                       "encoder", "decoder", "num_heads", "num_layers", "num_hidden", "residual", "sce_coef", "pooling",
                       "pre_train_model", "task_type",
                       "acc_mean", "acc_std"]
    elif args.filling_method == 'pcfi':
        header_list = ["filling_method", "downstream_model", "n_runs", "num_layers", "hidden_dim", "jk",
                       "alpha", "beta",
                       "acc_mean", "acc_std"]
    else:
        header_list = ["filling_method", "downstream_model", "n_runs", "num_layers", "hidden_dim", "jk",
                       "acc_mean", "acc_std"]
    # saving results
    with open(result_file_name, 'a+') as f:
        f.seek(0)
        val_header = f.read(7)
        if val_header != 'filling':
            dw = csv.DictWriter(f, delimiter=',', fieldnames=header_list)
            dw.writeheader()

        if args.filling_method == 'graphmae':
            line = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:.5f}, {:.5f}\n".format(
                args.filling_method, args.model, args.n_runs, args.num_layers, args.hidden_dim, args.jk,
                mae_args.encoder, mae_args.decoder, mae_args.num_heads, mae_args.num_layers, mae_args.num_hidden,
                mae_args.residual, mae_args.alpha_l, mae_args.pooling,
                mae_args.pretrained_model_name, mae_args.task_type,
                test_acc_mean, test_acc_std)
        elif args.filling_method == 'pcfi':
            line = "{}, {}, {}, {}, {}, {}, {}, {}, {:.5f}, {:.5f}\n".format(
                args.filling_method, args.model, args.n_runs, args.num_layers, args.hidden_dim, args.jk,
                args.alpha, args.beta,
                test_acc_mean, test_acc_std)
        else:
            line = "{}, {}, {}, {}, {}, {}, {:.5f}, {:.5f}\n".format(
                args.filling_method, args.model, args.n_runs, args.num_layers, args.hidden_dim, args.jk,
                test_acc_mean, test_acc_std)
        f.write(line)


def save_link_results(args, mae_args, test_auc_mean, test_auc_std, test_ap_mean, test_ap_std):
    result_dir = './results/'
    if not os.path.exists(result_dir):
        print("Creating directory for results ...")
        os.makedirs(result_dir)

    link_result_dir = os.path.join(result_dir, 'link_results/')
    if not os.path.exists(link_result_dir):
        print("Creating directory for link results ...")
        os.makedirs(link_result_dir)

    # parsing args
    result_file_name = (link_result_dir +
                        args.dataset_name + '_' +
                        args.filling_method + '_' +
                        args.mask_type + '_' +
                        str(args.missing_rate) + '.csv')

    if args.filling_method == 'graphmae':
        header_list = ["filling_method", "downstream_model", "n_runs", "hidden_dim", "jk",
                       "encoder", "decoder", "num_heads", "num_layers", "num_hidden", "residual", "sce_coef", "pooling",
                       "auc_mean", "auc_std", "ap_mean", "ap_std"]
    elif args.filling_method == 'pcfi':
        header_list = ["filling_method", "downstream_model", "n_runs", "hidden_dim", "jk",
                       "alpha", "beta",
                       "acc_mean", "acc_std", "ap_mean", "ap_std"]
    else:
        header_list = ["filling_method", "downstream_model", "n_runs", "hidden_dim", "jk",
                       "acc_mean", "acc_std", "ap_mean", "ap_std"]
    # saving results
    with open(result_file_name, 'a+') as f:
        f.seek(0)
        val_header = f.read(7)
        if val_header != 'filling':
            dw = csv.DictWriter(f, delimiter=',', fieldnames=header_list)
            dw.writeheader()

        if args.filling_method == 'graphmae':
            line = "{}, 'GAE', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                args.filling_method, args.n_runs, args.hidden_dim, args.jk,
                mae_args.encoder, mae_args.decoder, mae_args.num_heads, mae_args.num_layers, mae_args.num_hidden,
                mae_args.residual, mae_args.alpha_l, mae_args.pooling,
                test_auc_mean, test_auc_std, test_ap_mean, test_ap_std)
        elif args.filling_method == 'pcfi':
            line = "{}, 'GAE', {}, {}, {}, {}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                args.filling_method, args.n_runs, args.hidden_dim, args.jk,
                args.alpha, args.beta,
                test_auc_mean, test_auc_std, test_ap_mean, test_ap_std)
        else:
            line = "{}, 'GAE', {}, {}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                args.filling_method, args.n_runs, args.hidden_dim, args.jk,
                test_auc_mean, test_auc_std, test_ap_mean, test_ap_std)
        f.write(line)
