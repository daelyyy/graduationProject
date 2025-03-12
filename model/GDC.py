import time

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import scipy.sparse as sp
import numpy as np

# 定义 GDC 扩散过程
def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    start_time = time.time()  # 开始计时

    N = A.shape[0]

    # Step 1: 添加自环
    A_loop = sp.eye(N) + A

    # Step 2: 构建对称归一化的邻接矩阵
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec + 1e-10)  # 避免除零
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = (D_loop_invsqrt @ A_loop @ D_loop_invsqrt).tocsc()  # 转为 CSC 格式

    # Step 3: PPR-based 扩散
    I = sp.eye(N).tocsc()
    S = alpha * sp.linalg.inv(I - (1 - alpha) * T_sym)  # 确保 CSC 格式输入

    # Step 4: 稀疏化
    S_tilde = S.multiply(S >= eps)
    print(f"GDC: Non-zero elements after sparsification: {S_tilde.nnz}")

    # Step 5: 归一化
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / (D_tilde_vec + 1e-10)  # 避免除零

    elapsed_time = time.time() - start_time
    print(f"GDC execution time: {elapsed_time:.2f} seconds")

    return sp.csr_matrix(T_S)  # 返回 CSR 格式


# 自定义 GDC 模型
class GDC_Model(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GDC_Model, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.alpha = args.alpha
        self.eps = 0.0001
        self.hidden = args.hidden
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(f"Input feature shape: {x.shape}")

        # Step 1: 使用稀疏矩阵进行处理，避免直接转换为密集矩阵
        edge_index, edge_weight = dense_to_sparse(to_dense_adj(edge_index)[0])
        A_sparse = sp.csr_matrix((edge_weight.cpu().numpy(), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())), shape=(x.size(0), x.size(0)))

        # Step 2: 计算 GDC 扩散矩阵
        T_S = gdc(A_sparse, alpha=self.alpha, eps=self.eps)
        T_S_dense = torch.tensor(T_S.toarray(), dtype=torch.float32).to(x.device)

        # Step 3: 扩散特征
        x = torch.matmul(T_S_dense, x)

        # Step 4: MLP 层
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)