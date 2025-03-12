from torch import nn


class SGC_Net(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, dataset, args):
        super(SGC_Net, self).__init__()

        self.W = nn.Linear(dataset.num_features, dataset.num_classes)

    def forward(self, data):
        x = data.x
        return self.W(x)