import argparse
from typing import Tuple
from pathlib import Path
from timeit import default_timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader, Subset

from torch_geometric.data import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform


from dataset import ContactMapDataset


def train_valid_split(
    dataset: Dataset, split_pct: float = 0.8, method: str = "random", **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Creates training and validation DataLoaders from :obj:`dataset`.
    Parameters
    ----------
    dataset : Dataset
        A PyTorch dataset class derived from :obj:`torch.utils.data.Dataset`.
    split_pct : float
        Percentage of data to be used as training data after a split.
    method : str, default="random"
        Method to split the data. For random split use "random", for a simple
        partition, use "partition".
    **kwargs
        Keyword arguments to :obj:`torch.utils.data.DataLoader`. Includes,
        :obj:`batch_size`, :obj:`drop_last`, etc (see `PyTorch Docs
        <https://pytorch.org/docs/stable/data.html>`_).
    Raises
    ------
    ValueError
        If :obj:`method` is not "random" or "partition".
    """
    train_length = int(len(dataset) * split_pct)
    if method == "random":
        lengths = [train_length, len(dataset) - train_length]
        train_dataset, valid_dataset = random_split(dataset, lengths)
    elif method == "partition":
        indices = list(range(len(dataset)))
        train_dataset = Subset(dataset, indices[:train_length])
        valid_dataset = Subset(dataset, indices[train_length:])
    else:
        raise ValueError(f"Invalid method: {method}.")
    train_loader = DataLoader(train_dataset, **kwargs)
    valid_loader = DataLoader(valid_dataset, **kwargs)
    return train_loader, valid_loader


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
        net,
        aggr="add",
        root_weight=True,
        bias=True,
        **kwargs,
    ):
        super(NNConv_old, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = net
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
        reset(self.net)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.net(pseudo).view(-1, self.in_channels, self.out_channels)
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
        print("x", x.shape)
        print("edge_index", edge_index.shape)
        print("edge_attr", edge_attr.shape)
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--run_path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler_step", type=int, default=50)
    parser.add_argument("--scheduler_gamma", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--kernel_width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--node_features", type=int, default=20)
    parser.add_argument("--edge_features", type=int, default=6)
    parser.add_argument("--split_pct", type=float, default=0.8)
    args = parser.parse_args()

    # Validation of arguments
    if not args.data_path.exists():
        raise ValueError(f"data_path does not exist: {args.data_path}")

    # Make output directory
    args.run_path.mkdir()

    return args


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    avg_mse_loss = 0.0
    avg_l2_loss = 0.0
    for batch in train_loader:
        batch = batch["data"].to(device)

        optimizer.zero_grad()
        out = model(batch)
        print("out.shape:", out.shape)
        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1, 1))
        # mse.backward()
        loss = torch.norm(out.view(-1) - batch.y.view(-1), 1)
        loss.backward()

        l2 = loss_fn(out.view(args.batch_size, -1), batch.y.view(args.batch_size, -1))
        # l2.backward()

        optimizer.step()
        avg_mse_loss += mse.item()
        avg_l2_loss += l2.item()

    avg_l2_loss /= len(train_loader)
    avg_mse_loss /= len(train_loader)

    return avg_l2_loss, avg_mse_loss


def validate(model, valid_loader, loss_fn, device):
    model.eval()
    avg_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch["data"].to(device)
            out = model(batch)
            avg_loss += loss_fn(
                out.view(args.batch_size, -1), batch.y.view(args.batch_size, -1)
            ).item()
    avg_loss /= len(valid_loader)
    return avg_loss


def main():

    # Setup training and validation datasets
    dataset = ContactMapDataset(args.data_path)
    train_loader, valid_loader = train_valid_split(
        dataset,
        args.split_pct,
        method="partition",
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model, optimizer, loss function and scheduler
    model = KernelNN(
        args.width,
        args.kernel_width,
        args.depth,
        args.edge_features,
        args.node_features,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    loss_fn = LpLoss(size_average=False)

    # Start training
    best_loss = float("inf")
    for epoch in range(args.epochs):
        time = default_timer()
        avg_l2_train_loss, avg_mse_train_loss = train(
            model, train_loader, optimizer, loss_fn, device
        )
        avg_valid_loss = validate(model, valid_loader, loss_fn, device)
        scheduler.step()
        print(
            f"Epoch: {epoch}"
            f"\tTime: {default_timer() - time}"
            f"\tl2_train_loss: {avg_l2_train_loss}"
            f"\tmse_train_loss: {avg_mse_train_loss}"
            f"\tvalid_loss: {avg_valid_loss}"
        )

        # Save the model with the best validation loss
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, args.run_path / "best.pt")


if __name__ == "__main__":
    args = parse_args()
    main()
