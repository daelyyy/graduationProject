import torch
from torch.nn import Linear
import torch.nn.functional as F


class MLP_(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLP_, self).__init__()

        # 第一层：输入层到隐藏层
        self.lin1 = Linear(dataset.num_features, args.hidden)

        # 第二层：隐藏层到输出层
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        # Dropout率
        self.dropout = args.dropout

    def reset_parameters(self):
        # 重置参数
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x = data.x  # 获取节点特征

        # 第一层前向传播
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))

        # 第二层前向传播
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # 采用log softmax进行分类
        return F.log_softmax(x, dim=1)
