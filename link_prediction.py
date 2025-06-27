import networkx as nx
import torch
from torch_geometric.utils import k_hop_subgraph, subgraph
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from constants import DEVICE


def generate_subgraph_features(split_data, train_data, k_hops_list, additional_tqdm_msg='', disable_tqdm=False):
    assert len(k_hops_list) > 0
    all_subgraph_features_for_all_edges = []
    num_original_node_feat = train_data.x.shape[1]

    g_nx_train = nx.Graph()
    g_nx_train.add_nodes_from(range(train_data.num_nodes))
    g_nx_train.add_edges_from(train_data.edge_index.t().tolist())

    num_added_feat_per_k_hop = 6
    num_global_feat = 2
    total_num_added_feats = num_added_feat_per_k_hop * len(k_hops_list) + num_global_feat
    largest_k_hop = max(k_hops_list)

    tqdm_msg = 'generating subgraphs features'
    if additional_tqdm_msg:
        tqdm_msg += ': ' + additional_tqdm_msg
    for u_idx, v_idx in tqdm(split_data.edge_label_index.t(), desc=tqdm_msg, disable=disable_tqdm):
        u, v = u_idx.item(), v_idx.item()

        ############## global u_v features #####################
        adamic_adar_index = next(nx.adamic_adar_index(g_nx_train, [(u, v)]))[2]
        resource_alloc_index_val = next(nx.resource_allocation_index(g_nx_train, [(u, v)]))[2]
        uv_subgraph_features = torch.tensor([adamic_adar_index, resource_alloc_index_val], dtype=torch.float,
                                            device=DEVICE)
        ########################################################

        for k_hop in k_hops_list:
            ########## num nodes / edges in uv subgraph ################
            u_k_hop_neighbours, _, _, _ = k_hop_subgraph(u, k_hop, train_data.edge_index)
            v_k_hop_neighbours, _, _, _ = k_hop_subgraph(v, k_hop, train_data.edge_index)
            u_and_v_k_hop_neighbours = torch.cat([u_k_hop_neighbours, v_k_hop_neighbours]).unique().long()
            u_and_v_subgraph_edges, _ = subgraph(u_and_v_k_hop_neighbours, train_data.edge_index, relabel_nodes=False)
            num_nodes_in_u_v_subgraph = len(u_and_v_k_hop_neighbours)
            num_edges_in_u_v_subgraph = u_and_v_subgraph_edges.size(1)
            ################### mean features ##########################
            if num_nodes_in_u_v_subgraph > 0:
                mean_features_in_u_v_k_hop_subgraph = train_data.x[u_and_v_k_hop_neighbours].mean(dim=0)
            else:
                mean_features_in_u_v_k_hop_subgraph = torch.zeros(train_data.x.shape[1], device=DEVICE)
            ######################## shortest path #######################
            try:
                shortest_path = nx.shortest_path_length(g_nx_train, source=u, target=v)
            except nx.NetworkXNoPath:
                shortest_path = largest_k_hop + 1
            ####################### common neighbours #########################
            common_neighbors_num = len(list(nx.common_neighbors(g_nx_train, u, v)))
            ####################### jaccard_coef #########################
            neighbors_u = set(g_nx_train.neighbors(u))
            neighbors_v = set(g_nx_train.neighbors(v))
            union_neighbors = neighbors_u | neighbors_v
            jaccard_coef = 0 if len(union_neighbors) == 0 else len(neighbors_u & neighbors_v) / len(union_neighbors)

            ####################### density_k_subgraph_val #######################
            density_k_subgraph_val = (2 * num_edges_in_u_v_subgraph) / (num_nodes_in_u_v_subgraph * (
                        num_nodes_in_u_v_subgraph - 1)) if num_nodes_in_u_v_subgraph else 0.0
            ############################################################
            cur_k_hop_features = torch.cat([
                torch.tensor([
                    float(num_nodes_in_u_v_subgraph),
                    float(num_edges_in_u_v_subgraph),
                    float(shortest_path),
                    float(common_neighbors_num),
                    float(jaccard_coef),
                    float(density_k_subgraph_val),
                ], dtype=torch.float, device=DEVICE),
                mean_features_in_u_v_k_hop_subgraph
            ])
            uv_subgraph_features = torch.cat((uv_subgraph_features, cur_k_hop_features))

        all_subgraph_features_for_all_edges.append(uv_subgraph_features)

    if all_subgraph_features_for_all_edges:
        return torch.stack(all_subgraph_features_for_all_edges)
    else:
        return torch.empty(0, total_num_added_feats + num_original_node_feat, device=DEVICE)


def get_lp_logits(gnn_model, lp_model, split_data, with_additional_subgraph_feats, additional_subgraph_feats):
    z = gnn_model.encode(split_data.x, split_data.edge_index)
    edge_label_index = split_data.edge_label_index

    x_i = z[edge_label_index[0]]
    x_j = z[edge_label_index[1]]
    if with_additional_subgraph_feats:
        logits = lp_model(x_i, x_j, additional_subgraph_feats)
    else:
        logits = lp_model(x_i, x_j)
    return logits


def train_link_predictor(gnn_model, lp_model, split_data, optimizer, epochs=50,
                         additional_subgraph_feats=None, additional_tqdm_msg='', disable_tqdm=False):
    gnn_model.eval()
    lp_model.train()
    loss_arr = []

    with_additional_subgraph_feats = additional_subgraph_feats is not None

    for _ in tqdm(range(epochs),
                  desc=f'lp train{": " + additional_tqdm_msg if additional_tqdm_msg else additional_tqdm_msg}'):
        optimizer.zero_grad()

        logits = get_lp_logits(gnn_model, lp_model, split_data,
                               with_additional_subgraph_feats, additional_subgraph_feats)

        edge_label = split_data.edge_label
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), edge_label.float())
        loss.backward()
        optimizer.step()
        loss_arr.append(loss.item())
    return loss_arr


@torch.no_grad()
def test_link_predictor(gnn_model, lp_model, split_data, additional_subgraph_feats=None):
    gnn_model.eval()
    lp_model.eval()
    with_additional_subgraph_feats = additional_subgraph_feats is not None
    logits = get_lp_logits(gnn_model, lp_model, split_data, with_additional_subgraph_feats,
                           additional_subgraph_feats)
    edge_label = split_data.edge_label

    true_labels = edge_label.cpu().numpy()
    preds = torch.sigmoid(logits).view(-1).cpu().numpy()

    auc = roc_auc_score(true_labels, preds)
    return auc
