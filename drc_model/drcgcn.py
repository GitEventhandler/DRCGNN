import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class DRCGCLayer(nn.Module):
    def __init__(self, nhidden):
        super(DRCGCLayer, self).__init__()
        self.nhidden = nhidden
        self.next_x: torch.FloatTensor = None
        self.gamma = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.W = Parameter(torch.FloatTensor(nhidden, nhidden), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.zero_()
        self.W.data.zero_()

    def forward(self, A, X, I_1, beta):
        AX = torch.spmm(A, X)
        self.next_x = self.gamma * (X - AX)
        return torch.mm(AX, ((1.0 - beta) * I_1 + beta * self.W))


class DRCGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, tau=0.5, ):
        super(DRCGCN, self).__init__()
        self.nfeat = nfeat
        self.nlayers = nlayers
        self.nclass = nclass
        self.nhidden = nhidden
        self.tau = tau
        self.dropout = dropout
        self.init_dim_reduction = nn.Linear(nfeat, nhidden)
        self.I_1 = None
        self.convs = nn.ModuleList()
        for i in range(self.nlayers):
            layer = DRCGCLayer(self.nhidden)
            self.convs.append(layer)
        self.sorter = nn.Linear(self.nhidden, self.nclass)
        self.init_linear_params = list(self.init_dim_reduction.parameters())
        self.conv_params = list(self.convs.parameters())
        self.final_linear_params = list(self.sorter.parameters())
        self.reset_parameters()

    def reset_parameters(self):
        self.init_dim_reduction.reset_parameters()
        self.sorter.reset_parameters()

    def forward(self, X, A):
        if self.I_1 is None:
            self.I_1 = torch.eye(self.nhidden)
            self.I_1 = self.I_1.to(A.device)
        X = F.dropout(X, self.dropout, training=self.training)
        X = self.init_dim_reduction(X)
        outputs = torch.FloatTensor(A.shape[0], self.nhidden).to(A.device)
        outputs.data.fill_(0.0)
        outputs += X
        for i, conv_layer in enumerate(self.convs):
            X = F.dropout(X, self.dropout, self.training)
            outputs += conv_layer(A, X, self.I_1, self.tau / (i + 1))
            X = conv_layer.next_x
        return F.log_softmax(self.sorter(outputs), dim=1)
