import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
import numpy as np
import scipy.sparse as sp


class GNNsLF(nn.Module):
    def __init__(self, dataset, args):
        super(GNNsLF, self).__init__()

        # 线性层
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)

        # 传播层：使用LFPowerIteration或类似的传播机制
        adj = dataset[0].edge_index
        adj_sparse = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), (dataset.x.size(0), dataset.x.size(0)))
        self.propagation = LFPowerIteration(adj_sparse, alpha=args.alpha, mu=0.9, niter=args.K)

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


class LFPowerIteration(MessagePassing):
    def __init__(self, adj_matrix, alpha, mu, niter, drop_prob=0.0):
        super(LFPowerIteration, self).__init__(aggr='add')

        self.alpha = alpha
        self.mu = mu
        self.niter = niter

        # 预计算A_hat
        self.A_hat = self._calc_A_hat(adj_matrix)

        # Dropout设置
        self.dropout = MixedDropout(drop_prob)

    def reset_parameters(self):
        # 初始化传播的相关参数（如果需要）
        pass

    def _calc_A_hat(self, adj_matrix):
        nnodes = adj_matrix.shape[0]
        adj_matrix_dense = adj_matrix.to_dense()
        adj_matrix_sp = sp.csr_matrix(adj_matrix_dense.cpu().numpy())
        A = adj_matrix_sp + sp.eye(nnodes)
        D_vec = np.sum(A, axis=1).A1
        D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
        D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
        return D_invsqrt_corr @ A @ D_invsqrt_corr

    def forward(self, x, edge_index):
        # 传播步骤：每次迭代对输入进行传播
        x_detached = x.detach()  # 断开与计算图的关系
        # 将 scipy.sparse.csr_matrix 转换为 PyTorch 的稀疏 COO 张量
        row, col = self.A_hat.nonzero()
        indices = torch.LongTensor([row, col])  # row 和 col 是 scipy CSR 矩阵的行列索引
        values = torch.FloatTensor(self.A_hat.data)  # A.data 是存储的非零值
        shape = self.A_hat.shape  # 获取矩阵的形状

        # 转换为稀疏 COO 张量
        A_sparse = torch.sparse_coo_tensor(indices, values, shape)

        # 现在可以使用 torch.sparse.mm 进行矩阵乘法
        result = torch.sparse.mm(A_sparse, x)
        preds = (self.mu / (1 + self.alpha * self.mu - self.alpha)) * x_detached + (1 - self.mu) * torch.sparse.mm(
            A_sparse, x_detached)

        # 进行niter次迭代
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = (1 - 2 * self.alpha + self.mu * self.alpha) * A_drop @ preds + x

        return preds

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

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

