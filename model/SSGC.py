import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SSGCProp(MessagePassing):
    def __init__(self, K, alpha):
        super(SSGCProp, self).__init__(aggr='add')
        self.K = K  # Number of propagation steps
        self.alpha = alpha  # Balancing factor for self-loops

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute degree normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Initial propagation
        out = x
        for _ in range(self.K):
            out = self.propagate(edge_index, x=out, norm=norm) + out
        return self.alpha * x + (1 - self.alpha) * out / self.K

    def message(self, x_j, norm):
        # Weighted message passing
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        pass  # No parameters to reset


class SSGC_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(SSGC_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = SSGCProp(args.K, args.alpha)  # SSGC-specific propagation layer

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First linear transformation with dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second linear transformation
        x = self.lin2(x)

        # SSGC propagation
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
