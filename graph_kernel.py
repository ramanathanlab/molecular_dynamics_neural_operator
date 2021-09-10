import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform

from dataset import ContactMapDataset

from timeit import default_timer

import argparse
from pathlib import Path


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class NNConv_old(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),
    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        nn,
        aggr="add",
        root_weight=True,
        bias=True,
        **kwargs
    ):
        super(NNConv_old, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter("root", None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width ** 2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--split_pct",
        type=float,
        default=0.8,
        help="Percentage of data to use for training. The rest goes to validation.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default="./test_plots",
        help="Output directory for model results.",
    )
    args = parser.parse_args()
    return args


args = parse_args()

r = 4
s = int(((241 - 1) / r) + 1)
n = s ** 2
m = 100
k = 1

radius_train = 0.1
radius_test = 0.1

print("resolution", s)


ntrain = 100
ntest = 40

batch_size = 1
batch_size2 = 2
width = 64
ker_width = 1024
depth = 6
edge_features = 6
node_features = 6

epochs = 200
learning_rate = 0.0001
scheduler_step = 50
scheduler_gamma = 0.8

t1 = default_timer()

u_normalizer = GaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)

dataset = ContactMapDataset(args.data_path)
lengths = [
    int(len(dataset) * args.split_pct),
    int(len(dataset) * round(1 - args.split_pct, 2)),
]
train_dataset, valid_dataset = random_split(dataset, lengths)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=True, drop_last=True)

##################################################################################################

# training

##################################################################################################
t2 = default_timer()

print("preprocessing finished, time used:", t2 - t1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = KernelNN(width, ker_width, depth, edge_features, node_features).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=scheduler_step, gamma=scheduler_gamma
)

myloss = LpLoss(size_average=False)
u_normalizer.cuda()

model.train()
ttrain = np.zeros((epochs,))
ttest16 = np.zeros((epochs,))
ttest31 = np.zeros((epochs,))
ttest61 = np.zeros((epochs,))

for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0.0
    train_l2 = 0.0
    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1, 1))
        # mse.backward()
        loss = torch.norm(out.view(-1) - batch.y.view(-1), 1)
        loss.backward()

        l2 = myloss(
            u_normalizer.decode(out.view(batch_size, -1)),
            u_normalizer.decode(batch.y.view(batch_size, -1)),
        )
        # l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()

    model.eval()

    ttrain[ep] = train_l2 / (ntrain * k)

    print(ep, " time:", t2 - t1, " train_mse:", train_mse / len(train_loader))

t1 = default_timer()
u_normalizer.cpu()
model = model.cpu()
test_l2_16 = 0.0
test_l2_31 = 0.0
test_l2_61 = 0.0
model.eval()
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        test_l2_16 += myloss(
            u_normalizer.decode(out.view(batch_size2, -1)),
            batch.y.view(batch_size2, -1),
        )
ttest61[ep] = test_l2_61 / ntest
t2 = default_timer()

print(
    " time:",
    t2 - t1,
    " train_mse:",
    train_mse / len(train_loader),
    " test16:",
    test_l2_16 / ntest,
    " test31:",
    test_l2_31 / ntest,
    " test61:",
    test_l2_61 / ntest,
)
np.savetxt(path_train_err + ".txt", ttrain)
np.savetxt(path_test_err16 + ".txt", ttest16)
np.savetxt(path_test_err31 + ".txt", ttest31)
np.savetxt(path_test_err61 + ".txt", ttest61)

torch.save(model, path_model)
