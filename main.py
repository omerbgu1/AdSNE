import torch
from openai.types.beta.assistant_stream_event import ThreadRunRequiresAction
from tqdm import tqdm
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS, Actor
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import k_hop_subgraph, subgraph
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import tempfile
from pathlib import Path

from v2.constants import SEED, DEVICE

from v2.link_prediction import train_link_predictor, generate_subgraph_features, test_link_predictor
from v2.models import GCN, LinkPredictor
from v2.node_classification import train_node_classifier, test_node_classifier
from v2.os_utils import set_seed
from v2.plots import plot_lp_loss, plot_lp_auc


def load_dataset_custom(dataset_name):
    root_path = f'/tmp/{dataset_name}'
    if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        return Planetoid(root_path, name=dataset_name)
    elif dataset_name == "CoauthorCS":
        return Coauthor(root_path, name="CS")
    elif dataset_name == "CoauthorPhysics":
        return Coauthor(root_path, name="Physics")
    elif dataset_name == "AmazonComputers":
        return Amazon(root_path, name="Computers")
    elif dataset_name == "AmazonPhoto":
        return Amazon(root_path, name="Photo")
    elif dataset_name == "WikiCS":
        return WikiCS(root_path)
    elif dataset_name == "Actor":
        return Actor(root_path)
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")


def main_algo(dataset_name, k_hop, disable_tqdm=False):
    nodes_class_epochs = 100
    k_hops_list = [k_hop]
    lp_epochs = 500
    lp_hidden_channels = 128
    embedded_dim = 32

    dataset = load_dataset_custom(dataset_name)
    data = dataset[0].to(DEVICE)
    data.x = data.x.to(torch.float)
    node_features_num = dataset.num_node_features

    ############ node classification (############
    if not hasattr(data, 'train_mask'):  # CoauthorPhysics, AmazonComputers, AmazonPhoto, CoauthorCS
        splitter = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
        data = splitter(data)
    gnn_model = GCN(node_features_num, 64, embedded_dim, dataset.num_classes).to(DEVICE)
    optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)

    train_node_classifier(gnn_model, data, optimizer_gnn, nodes_class_epochs, disable_tqdm)
    node_accuracy = test_node_classifier(gnn_model, data)

    ############ link prediction (2-hop) ############
    link_split = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, neg_sampling_ratio=1.0)
    train_lp_data, val_lp_data, test_lp_data = link_split(data)

    train_subgraph_feats = generate_subgraph_features(train_lp_data, train_lp_data, k_hops_list, 'train',
                                                      disable_tqdm=disable_tqdm).to(DEVICE)
    test_subgraph_feats = generate_subgraph_features(test_lp_data, train_lp_data, k_hops_list, 'test',
                                                     disable_tqdm=disable_tqdm).to(DEVICE)
    num_of_lp_features_and_added = train_subgraph_feats.shape[1]

    lp_baseline_model = LinkPredictor(embedded_dim, lp_hidden_channels).to(DEVICE)
    optimizer_lp_baseline = torch.optim.Adam(lp_baseline_model.parameters(), lr=0.01)

    lp_baseline_loss_arr = train_link_predictor(gnn_model, lp_baseline_model, train_lp_data, optimizer_lp_baseline,
                                                epochs=lp_epochs, additional_tqdm_msg='baseline',
                                                disable_tqdm=disable_tqdm)
    lp_baseline_auc = test_link_predictor(gnn_model, lp_baseline_model, test_lp_data)

    ############ link prediction with subgraph aware ############
    lp_subgraph_aware_model = LinkPredictor(embedded_dim, lp_hidden_channels, num_of_lp_features_and_added).to(DEVICE)
    lp_subgraph_aware_optim = torch.optim.Adam(lp_subgraph_aware_model.parameters(), lr=0.01)

    lp_subgraph_aware_loss_arr = train_link_predictor(gnn_model, lp_subgraph_aware_model, train_lp_data,
                                                      lp_subgraph_aware_optim,
                                                      epochs=lp_epochs, additional_subgraph_feats=train_subgraph_feats,
                                                      additional_tqdm_msg='subgraph aware', disable_tqdm=disable_tqdm)
    lp_subgraph_aware_auc = test_link_predictor(gnn_model, lp_subgraph_aware_model, test_lp_data,
                                                additional_subgraph_feats=test_subgraph_feats)

    return lp_epochs, lp_baseline_loss_arr, lp_subgraph_aware_loss_arr, lp_baseline_auc, lp_subgraph_aware_auc, node_accuracy


def main():
    datasets_names = [
        "Cora",
        "CiteSeer",
        "PubMed",  # 70k
        # 'CoauthorCS'  # 130K
        # 'CoauthorPhysics',  # 400k
        # 'AmazonComputers',  # 400k
        # 'AmazonPhoto',  # 200k
        # 'Actor',  # error
        # 'WikiCS',  # error
    ]
    num_repetition = 1
    outer_loop_tqdm = False
    k_hops_list = [2, 3]
    for dataset_name in datasets_names:
        lp_subgraph_aware_auc_list = []
        for k_hop in k_hops_list:
            set_seed(SEED)
            (
                lp_baseline_loss_arr, lp_subgraph_aware_loss_arr,
                lp_baseline_auc, lp_subgraph_aware_auc, node_accuracy
            ) = [[] for _ in range(5)]
            for _ in tqdm(range(num_repetition), disable=not outer_loop_tqdm):
                (
                    lp_epochs, cur_lp_baseline_loss_arr, cur_lp_subgraph_aware_loss_arr,
                    cur_lp_baseline_auc, cur_lp_subgraph_aware_auc, cur_node_accuracy
                ) = main_algo(dataset_name, k_hop, disable_tqdm=outer_loop_tqdm)

                lp_baseline_loss_arr.append(cur_lp_baseline_loss_arr)
                lp_subgraph_aware_loss_arr.append(cur_lp_subgraph_aware_loss_arr)
                lp_baseline_auc.append(cur_lp_baseline_auc)
                lp_subgraph_aware_auc.append(cur_lp_subgraph_aware_auc)
                node_accuracy.append(cur_node_accuracy)
            lp_baseline_loss_arr = np.array(lp_baseline_loss_arr).mean(axis=0)
            lp_subgraph_aware_loss_arr = np.array(lp_subgraph_aware_loss_arr).mean(axis=0)
            lp_baseline_auc = np.mean(lp_baseline_auc)
            lp_subgraph_aware_auc = np.mean(lp_subgraph_aware_auc)
            lp_subgraph_aware_auc_list.append(lp_subgraph_aware_auc)
            node_accuracy = np.mean(node_accuracy)

            plot_lp_loss(lp_epochs, lp_baseline_loss_arr, lp_subgraph_aware_loss_arr, dataset_name, k_hop)
        plot_lp_auc(node_accuracy, lp_baseline_auc, lp_subgraph_aware_auc_list, dataset_name, k_hops_list)
    return


if __name__ == "__main__":
    main()
