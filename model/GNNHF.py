import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
import numpy as np
import scipy.sparse as sp


class GNNsHF(nn.Module):
    def __init__(self, dataset, args):
        super(GNNsHF, self).__init__()

        # 线性层
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)

        # 传播层：使用LFPowerIteration或类似的传播机制
        adj = dataset[0].edge_index
        adj_sparse = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), (dataset.x.size(0), dataset.x.size(0)))
        self.propagation = HFPowerIteration(adj_sparse, alpha=args.alpha, beta=0.9, niter=args.K)

        # Dropout和其他参数
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.propagation.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一层线性变换和激活
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        # 第二层线性变换
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # 传播
        if self.dropout == 0.0:
            x = self.propagation(x, edge_index)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagation(x, edge_index)

        return F.log_softmax(x, dim=1)



class HFPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, beta: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.niter = niter

        M = self._calc_A_hat(adj_matrix)
        nnodes = adj_matrix.shape[0]
        L = sp.eye(nnodes) - M
        self.register_buffer('L_hat', sparse_matrix_to_torch(L)) # L
        self.register_buffer('A_hat', sparse_matrix_to_torch(((alpha * beta + 1 - alpha)/(alpha*beta + 1))* M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        #  Z_0 = 1/(alpha*beta + 1) H + beta/(alpha*beta + 1) LH
        preds = 1/(self.alpha * self.beta + 1) * local_preds + (self.beta/(self.alpha * self.beta + 1)) * self.L_hat  @ local_preds
        local_preds = self.alpha * preds # residual part: alpha/(alpha*beta + 1) H + alpha * beta/(alpha*beta + 1) LH
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + local_preds
        return preds


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def _calc_A_hat(self, adj_matrix):
        nnodes = adj_matrix.shape[0]
        adj_matrix_dense = adj_matrix.to_dense()
        adj_matrix_sp = sp.csr_matrix(adj_matrix_dense.cpu().numpy())
        A = adj_matrix_sp + sp.eye(nnodes)
        D_vec = np.sum(A, axis=1).A1
        D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
        D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
        return D_invsqrt_corr @ A @ D_invsqrt_corr

    def reset_parameters(self):
        # 初始化传播的相关参数（如果需要）
        pass

def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)

class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if isinstance(input, (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix)):
            input = sparse_matrix_to_torch(input)  # 将 scipy.sparse 转为 PyTorch 稀疏张量

        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)

