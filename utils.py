import os
import pickle as pkl
import random
import sys
from csbm.csbm_dataset import SynCSBM

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

PROJECT_ROOT = os.path.split(__file__)[0]


def _adj_single_side_norm(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -1.0).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat.dot(adj).tocoo()


def _parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _get_sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def _preprocess_features(features):
    row_sum = np.array(features.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    r_inv = np.power(row_sum.astype(np.float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def _load_citation(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(PROJECT_ROOT + "/dataset/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file(
        PROJECT_ROOT + "/dataset/data/ind.{}.test.index".format(dataset_str)
    )
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = _get_sample_mask(idx_train, labels.shape[0])
    val_mask = _get_sample_mask(idx_val, labels.shape[0])
    test_mask = _get_sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask


def load_data(dataset_name, splits_file_path=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = _load_citation(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join(
            PROJECT_ROOT + '/dataset/raw_data',
            dataset_name,
            'out1_graph_edges.txt'
        )
        graph_node_features_and_labels_file_path = os.path.join(
            PROJECT_ROOT + "/dataset/raw_data",
            dataset_name,
            'out1_node_feature_label.txt'
        )

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])]
        )
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])]
        )
    features = _preprocess_features(features)
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    if dataset_name == 'film' or dataset_name == 'texas' or dataset_name == 'wisconsin' or dataset_name == 'cornell':
        adj = _adj_single_side_norm(adj)
    else:
        s = adj.sum(1)
        adj = adj / np.tile(s, s.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels


def load_csbm(phi):
    assert phi in [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0], "Phi is limited to -1, 1, 0.25."
    dataset_name = "cSBM_phi_" + '{:.2f}'.format(phi)
    root = PROJECT_ROOT + '/dataset/csbm'
    dataset = SynCSBM(root=root, name=dataset_name)
    data = dataset.data
    graph = nx.DiGraph()
    for i in range(data.num_nodes):
        graph.add_node(i, features=data.x[i].numpy(), label=int(data.y[i]))
    for i in range(data.edge_index.shape[1]):
        src, dst = int(data.edge_index[0][i]), int(data.edge_index[1][i])
        graph.add_edge(src, dst)
    adj = nx.adjacency_matrix(graph, sorted(graph.nodes()))
    features = np.array(
        [features for _, features in sorted(graph.nodes(data='features'), key=lambda x: x[0])]
    )
    labels = np.array(
        [label for _, label in sorted(graph.nodes(data='label'), key=lambda x: x[0])]
    )
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    adj = _adj_single_side_norm(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
