import numpy as np
from torch_geometric.data import Data
import pickle
from datetime import datetime
import os.path as osp
import os
import argparse
from torch_geometric.data import InMemoryDataset
import torch

DATASET_ROOT = os.path.split(__file__)[0] + '/dataset/csbm'


def _index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def _random_splits(data, num_classes,train_per_class, val_per_class ):
    class_indexes = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        class_indexes.append(index)
    train_index = torch.cat([i[0:train_per_class] for i in class_indexes], dim=0)
    validate_index = torch.cat([i[train_per_class:train_per_class + val_per_class] for i in class_indexes], dim=0)
    test_index = torch.cat([i[train_per_class + val_per_class:] for i in class_indexes], dim=0)
    data.train_mask = _index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = _index_to_mask(validate_index, size=data.num_nodes)
    data.test_mask = _index_to_mask(test_index, size=data.num_nodes)
    return data


def _context_sbm(n, d, Lambda, p, mu, train_percent=0.6):
    c_in = d + np.sqrt(d) * Lambda
    c_out = d - np.sqrt(d) * Lambda
    y = np.ones(n)
    y[int(n / 2) + 1:] = -1
    y = np.asarray(y, dtype=int)

    edge_index = [[], []]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[i] * y[j] > 0:
                Flip = np.random.binomial(1, c_in / n)
            else:
                Flip = np.random.binomial(1, c_out / n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)
    x = np.zeros([n, p])
    u = np.random.normal(0, 1 / np.sqrt(p), [1, p])
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu / n) * y[i] * u + Z / np.sqrt(p)
    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))
    data.coalesce()

    num_class = len(np.unique(y))
    val_lb = int(n * train_percent)
    percls_trn = int(round(train_percent * n / num_class))
    data = _random_splits(data, num_class, percls_trn, val_lb)
    data.Lambda = Lambda
    data.mu = mu
    data.n = n
    data.p = p
    data.d = d
    data.train_percent = train_percent

    return data


def _parameterized_lambda_and_mu(theta, p, n, epsilon=0.1):
    from math import pi
    gamma = n / p
    assert (theta >= -1) and (theta <= 1)
    Lambda = np.sqrt(1 + epsilon) * np.sin(theta * pi / 2)
    mu = np.sqrt(gamma * (1 + epsilon)) * np.cos(theta * pi / 2)
    return Lambda, mu


def _save_data_to_pickle(data, p2root, file_name=None):
    now = datetime.now()
    surfix = now.strftime('%b_%d_%Y-%H:%M')
    if file_name is None:
        tmp_data_name = '_'.join(['cSBM_data', surfix])
    else:
        tmp_data_name = file_name
    p2cSBM_data = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2cSBM_data, 'bw') as f:
        pickle.dump(data, f)
    return p2cSBM_data


class SynCSBM(InMemoryDataset):
    def __init__(self, root, name=None,
                 n=1000, d=10, p=1500, Lambda=None, mu=None,
                 epsilon=0.1, theta=0.5,
                 train_ratio=0.6,
                 transform=None, pre_transform=None):

        now = datetime.now()
        suffix = now.strftime('%b_%d_%Y-%H:%M')
        if name is None:
            self.name = '_'.join(['cSBM_data', suffix])
        else:
            self.name = name
        self._n = n
        self._d = d
        self._p = p
        self._Lambda = Lambda
        self._mu = mu
        self._epsilon = epsilon
        self._theta = theta
        self._train_percent = train_ratio
        root = osp.join(root, self.name)
        if not osp.isdir(root):
            os.makedirs(root)
        super(SynCSBM, self).__init__(
            root, transform, pre_transform
        )
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.Lambda = self.data.Lambda.item()
        self.mu = self.data.mu.item()
        self.n = self.data.n.item()
        self.p = self.data.p.item()
        self.d = self.data.d.item()
        self.train_percent = self.data.train_percent.item()

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise Exception("Dataset files not exists.")

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
