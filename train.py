import argparse
import time
import uuid

import torch.nn.functional as F
import torch.optim as optim

from drc_model.drcgcn import DRCGCN
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--wd1', type=float, default=5e-2, help='Weight decay (L2 loss on each gcn layer\'s parameters).')
parser.add_argument('--wd2', type=float, default=5e-3, help='Weight decay (L2 loss on init fc layer\'s parameters).')
parser.add_argument('--wd3', type=float, default=5e-3, help='Weight decay (L2 loss on final fc layer\'s parameters).')
parser.add_argument('--tau', type=float, default=0.5, help='Tau value.')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=-1, help='Hidden layer\'s dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience for early stop.')
parser.add_argument('--data', default='cora', help='Dataset to use.')
parser.add_argument('--devid', type=int, default=0, help='Device id, A integer.')
parser.add_argument('--phi', type=float, default=0.0, help='Only work for cSBM dataset, from -1 to 1 step 0.25.')
parser.add_argument('--index', type=int, default=-1, help='Which split to running on.')
parser.add_argument('--nolog', action='store_true', default=False, help='Invoke to prevent training log from output.')
args = parser.parse_args()
set_seed(args.seed)
cuda_id = "cuda:" + str(args.devid)
device = torch.device(cuda_id)
checkpoint_file = PROJECT_ROOT + '/pretrained/full-' + args.data + '-' + uuid.uuid4().hex + '.pt'
if args.data == 'csbm':
    print(cuda_id, checkpoint_file, "on dataset", args.data, '(Φ={:.2f})'.format(args.phi))
else:
    print(cuda_id, checkpoint_file, "on dataset", args.data)
if not os.path.exists(PROJECT_ROOT + '/pretrained'):
    os.makedirs(PROJECT_ROOT + '/pretrained')


def train_step(model, optimizer, features, labels, adj, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate_step(model, features, labels, adj, idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test_step(model, features, labels, adj, idx_test):
    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


def train(dataset_name, split_index=0):
    if dataset_name == "csbm":
        adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = load_csbm(
            args.phi
        )
    else:
        adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = load_data(
            dataset_name,
            split_index
        )
    features = features.to(device)
    adj = adj.to(device)
    model = DRCGCN(
        nfeat=features.shape[1],
        nlayers=args.layer,
        nhidden=args.hidden,
        nclass=int(labels.max()) + 1,
        dropout=args.dropout,
        tau=args.tau
    ).to(device)
    optimizer = optim.Adam(
        [
            {'params': model.conv_params, 'weight_decay': args.wd1},
            {'params': model.init_linear_params, 'weight_decay': args.wd2},
            {'params': model.final_linear_params, 'weight_decay': args.wd3},
        ],
        lr=args.lr
    )
    bad_counter = 0
    loss_best = 100
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train_step(model, optimizer, features, labels, adj, idx_train)
        loss_val, acc_val = validate_step(model, features, labels, adj, idx_val)
        if not args.nolog:
            log_message = '-   Epoch:{:04d}'.format(epoch + 1) + ' train' + ' loss:{:.3f}'.format(
                loss_tra) + ' acc:{:.2f}'.format(acc_tra * 100) + ' | val' + ' loss:{:.3f}'.format(
                loss_val) + ' acc:{:.2f}'.format(acc_val * 100)
            print('\b' * len(log_message), end='')
            print(log_message, end='')
        if loss_val < loss_best or epoch == 0:
            loss_best = loss_val
            torch.save(model.state_dict(), checkpoint_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    acc = test_step(model, features, labels, adj, idx_test)[1]
    return acc


if __name__ == '__main__':
    if args.data in ["cora", "pubmed", "citeseer", "film", "chameleon", "squirrel"]:
        t_total = time.time()
        if 0 <= args.index and args.index <= 9:
            i = args.index
            dataset = args.data
            split_npz_path = PROJECT_ROOT + '/dataset/splits/' + args.data + '_split_0.6_0.2_' + str(i) + '.npz'
            acc = train(dataset, split_npz_path)
            print('')
            print("Test accuracy", ": {:.2f}".format(acc * 100))
        else:
            acc_list = []
            for i in range(10):
                dataset = args.data
                split_npz_path = PROJECT_ROOT + '/dataset/splits/' + args.data + '_split_0.6_0.2_' + str(i) + '.npz'
                acc_list.append(train(dataset, split_npz_path) * 100)
                print('')
                print("Index", i, "test accuracy", ": {:.2f}".format(acc_list[-1]))
            print("Result on dataset", args.data)
            print("Max: {:.2f}".format(max(acc_list)), "Min: {:.2f}".format(min(acc_list)))
            print("All: ", ', '.join(['{:.2f}'.format(i) for i in acc_list]))
            print("Train cost: {:.4f}s".format(time.time() - t_total))
            print("Test acc: {:.2f}".format(np.mean(acc_list)))
    elif args.data in ["csbm"]:
        acc = train("csbm")
        print('')
        print("Accuracy for cSBM (Φ={:.2f}) : {:.2f}".format(args.phi, acc * 100))
    else:
        raise Exception(
            "Choose from %s" % ','.join(["cora", "pubmed", "citeseer", "film", "chameleon", "squirrel", "csbm"]))
