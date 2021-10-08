import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
from pathlib import Path
from timeit import default_timer
from collections import defaultdict
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader, Subset

from torch_geometric.data import DataLoader
from torch_geometric.loader import DataListLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.nn import DataParallel

import wandb
import os
import imageio
import pdb
import sys

from dataset import ContactMapDataset, PairData, ContactMapNewDataset

from mdlearn.utils import log_latent_visualization

EPS = 1e-15


def train_valid_split(
        dataset: Dataset, split_pct: float = 0.8, method: str = "random", 
        augment_by_reversing_prob: float = 0,
        augment_by_rotating180_prob: float = 0,
        augment_by_translating_prob: float = 0,
        augment_with_noise_mu: float = 0,
        **kwargs
) -> Tuple[DataListLoader, DataListLoader]:
    """Creates training and validation DataLoaders from :obj:`dataset`.
    Parameters
    ----------
    dataset : 
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
    train_loader = DataListLoader(train_dataset, **kwargs)
    valid_loader = DataListLoader(valid_dataset, **kwargs)
    return train_loader, valid_loader, train_dataset, valid_dataset


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
        print(f'NNConv_old kernel: {self.net}')
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
    def __init__(
            self,
            width: int,
            ker_width: int,
            depth: int,
            ker_in: int,
            in_width: int = 1,
            out_width: int = 1,
            num_embeddings: int = 20,
            embedding_dim: int = 4,
            x_position_dim: int = 3,
            num_nodes: int = 28,
            window_size: int = 10
    ) -> None:
        super(KernelNN, self).__init__()
        self.depth = depth
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.x_position_dim = x_position_dim
        self.num_nodes = num_nodes
        self.lstm = nn.LSTM(x_position_dim, x_position_dim)
        self.lstm_fc = torch.nn.Linear(x_position_dim, x_position_dim)
        self.window_size = window_size
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width ** 2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        self.conv2 = NNConv_old(width, width, kernel, aggr="mean")

        self.fc2 = torch.nn.Linear(width, out_width)

    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = data.x_position.reshape(-1, self.window_size, self.num_nodes, 3)
        x = torch.swapaxes(x, 0, 1)
        # process the window of previous frames
        hidden = (torch.randn(1, self.num_nodes, 3).cuda(),
                  torch.randn(1, self.num_nodes, 3).cuda())
        for i in x:
            #print(x.shape)
            x, hidden = self.lstm(i, hidden)
        x = self.lstm_fc(x)
        #print('xc', x.shape)
        # Use an embedding layer to map the onehot aminoacid vector to
        # a dense vector and then concatenate the result with the positions
        # emb = self.emb(data.x_aminoacid.view(args.batch_size, -1, self.num_embeddings))
        emb = self.emb(data.x_aminoacid)
        x = x.reshape(emb.shape[0], -1)
        # print("data.x_aminoacid", data.x_aminoacid.shape)
        # print("data.x_position:", data.x_position.shape)
        x = torch.cat((emb, x), dim=1)
        #print('xemb', x.shape)
        x = F.relu(self.fc1(x))
        for k in range(self.depth):
            #print(k, x.shape)
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        for k in range(self.depth):
            #print(k, x.shape)
            x = F.relu(self.conv2(x, edge_index, edge_attr))
        if return_latent:
            latent_dim = torch.clone(x)
        x = self.fc2(x)
        #print('xf', x.shape)
        #sys.exit(0)
        if return_latent:
            return [x, latent_dim]
        else:
            return x

# TODO: 
# evaluate GNN separately.. may be keep all 10 frames till the very end
# modify edge and node features
# use union of edges from all 10 frames (or use all 28^2 edges?)
# Skip LSTM and do GNN on 10 frame data?
# 

class KernelNNFixed(torch.nn.Module):
    def __init__(
            self,
            width: int,
            ker_width: int,
            depth: int,
            ker_in: int,
            in_width: int = 1,
            out_width: int = 1,
            num_embeddings: int = 20,
            embedding_dim: int = 4,
            x_position_dim: int = 3,
            num_nodes: int = 28,
            window_size: int = 10,
            lstm_num_layers: int = 1 #TODO: make it a param
    ) -> None:
        super(KernelNNFixed, self).__init__()
        self.depth = depth
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.x_position_dim = x_position_dim
        self.num_nodes = num_nodes
        self.lstm = nn.LSTM(num_nodes * x_position_dim, num_nodes * x_position_dim, num_layers=lstm_num_layers)
        #self.lstm_fc = torch.nn.Linear(x_position_dim, x_position_dim)
        self.window_size = window_size
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        kernel = DenseNet([ker_in, ker_width, ker_width, width ** 2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        self.conv2 = NNConv_old(width, width, kernel, aggr="mean")
        self.fc0 = torch.nn.Linear(num_nodes * x_position_dim, num_nodes * x_position_dim) #TODO: could expand to num_frames
        #self.fc1 = torch.nn.Linear(x_position_dim * window_size, x_position_dim)
        self.fc1 = torch.nn.Linear(window_size * num_nodes * x_position_dim, num_nodes * x_position_dim) #width)
        #self.fc1 = torch.nn.Linear(x_position_dim * window_size, x_position_dim)

        self.fc2 = torch.nn.Linear(width + embedding_dim, width)
        self.fc3 = torch.nn.Linear(num_nodes * x_position_dim, num_nodes * x_position_dim)#width, out_width)
        self.lstm_num_layers = lstm_num_layers

    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = data.x_position #.reshape(-1, self.window_size, self.num_nodes, 3)
        #print('x0', x.shape) # x0 torch.Size([2560, 28, 3])
        x = x.reshape(-1, self.window_size, x.shape[-2] * x.shape[-1])
        #print('x1', x.shape) # x1 torch.Size([256, 10, 84])
        x = torch.swapaxes(x, 0, 1)
        #print('x2', x.shape) # x2 torch.Size([10, 256, 84])
        x = self.fc0(x)
        #print('x3', x.shape) # x3 torch.Size([10, 256, 84])
        self.lstm.flatten_parameters()
        hidden = (
            torch.zeros(self.lstm_num_layers, x.shape[-2], x.shape[-1]).cuda(),
            torch.zeros(self.lstm_num_layers, x.shape[-2], x.shape[-1]).cuda()
        )          
        x, hidden = self.lstm(x, hidden)
        #print('x3_', x.shape) # x3_ torch.Size([10, 256, 84])

        if True:
            #print('x3b', x.shape)
            x = torch.swapaxes(x, 0, 1) # 256, 10, 84
            #print('x3c', x.shape)
            x = x.reshape(x.shape[0], x.shape[1], -1, self.x_position_dim) # 256, 10, 28, 3
            #print('x3d', x.shape)
            x = torch.swapaxes(x, 1, 2) # 256, 28, 10, 3
            #print('x3e', x.shape)
            x = x.reshape(-1, self.x_position_dim * self.window_size)
            #print('x3f', x.shape) # x3f torch.Size([7168, 30])
            #x = self.fc1(x)
            #print('x4', x.shape)
        if False:
            emb = self.emb(data.x_aminoacid)
            x = x.reshape(emb.shape[0], -1)
            #print('x4b', x.shape) # x4b torch.Size([7168, 128])
            # print("data.x_aminoacid", data.x_aminoacid.shape)
            # print("data.x_position:", data.x_position.shape)
            x = torch.cat((emb, x), dim=1)
            #print('xemb', x.shape) # xemb torch.Size([7168, 132])
            x = torch.swapaxes(x, 0, 1).reshape(-1, self.window_size * self.x_position_dim * self.num_nodes)
            x = self.fc1(x) #F.relu(x)) # reduces 840 to 84
            #x = F.relu(self.fc2(x))

        if True:
            for k in range(self.depth):
                #print(k, x.shape)
                x = F.relu(self.conv1(x, edge_index, edge_attr))
            for k in range(self.depth):
                #print(k, x.shape)
                x = F.relu(self.conv2(x, edge_index, edge_attr))
            if return_latent:
                latent_dim = torch.clone(x)
        #print('xf_', x.shape) # xf_ torch.Size([256, 84])
        x = self.fc3(x) #F.relu(x))
        #print('xf__', x.shape)
        x = x.reshape(-1, self.x_position_dim)
        #print('xf', x.shape)
        if return_latent:
            return [x, latent_dim]
        else:
            return x


class BasicGNN(torch.nn.Module):
    def __init__(
            self,
            width: int,
            ker_width: int,
            depth: int,
            ker_in: int,            
            in_width: int = 1, # total number of node features: 3 coords x 10 frames + 4 embedding
            out_width: int = 1,
            num_embeddings: int = 20,
            embedding_dim: int = 4,
            x_position_dim: int = 3,
            num_nodes: int = 28,
            window_size: int = 10
    ) -> None:
        super(BasicGNN, self).__init__()
        self.depth = depth
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.x_position_dim = x_position_dim
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.fc1 = torch.nn.Linear(in_width, width)
        kernel = DenseNet([ker_in, ker_width, ker_width, width ** 2], torch.nn.ReLU)
        #kernel = DenseNet([ker_in, width ** 2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        #self.conv2 = NNConv_old(width, width, kernel, aggr="mean")
        #self.fcd = torch.nn.Linear(width, width)

        self.fc2 = torch.nn.Linear(width, out_width)

    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = data.x_position.reshape(-1, self.window_size, self.num_nodes, 3)
        x = torch.swapaxes(x, 1, 2)
        emb = self.emb(data.x_aminoacid)
        x = x.reshape(emb.shape[0], -1)
        x = torch.cat((emb, x), dim=1)
        x = F.relu(self.fc1(x))
        for k in range(self.depth):
            #x = self.fcd(F.relu(self.conv1(x, edge_index, edge_attr)))
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        #for k in range(self.depth):
        #    x = F.relu(self.conv2(x, edge_index, edge_attr))
        if return_latent:
            latent_dim = torch.clone(x)
        x = self.fc2(x)
        if return_latent:
            return [x, latent_dim]
        else:
            return x


class BasicLSTM(torch.nn.Module):
    def __init__(
            self,
            x_position_dim: int = 3,
            num_layers: int = 10, # 2,3,4,5,10 produce roughly same result; 1 is worse (for frame_step 100). For frame_step=1, num_layers=1 is best
            window_size: int = 10,
    ) -> None:
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=x_position_dim, hidden_size=x_position_dim, num_layers=num_layers)
        self.fc0 = torch.nn.Linear(x_position_dim, x_position_dim)
        self.fc1 = torch.nn.Linear(x_position_dim * window_size, x_position_dim)
        self.num_layers = num_layers
        self.x_position_dim = x_position_dim
        self.window_size = window_size


    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        x = data.x_position
        #x = x.reshape(-1, args.window_size, 1, x.shape[-2] * x.shape[-1])
        x = x.reshape(-1, self.window_size, x.shape[-2] * x.shape[-1])
        x = torch.swapaxes(x, 0, 1)
        x = self.fc0(x)
        self.lstm.flatten_parameters()
        #for x_i in x:
        #    xx, hidden = self.lstm(x_i, hidden)
        #print('x', x.shape)
        hidden = (torch.zeros(self.num_layers, x.shape[-2], x.shape[-1]).cuda(),
                  torch.zeros(self.num_layers, x.shape[-2], x.shape[-1]).cuda())
        xx, hidden = self.lstm(x, hidden)
        #print('xx', xx.shape)
        x = self.fc1(torch.swapaxes(xx, 0, 1).reshape(-1, self.x_position_dim * self.window_size))
        x = x.reshape(-1, 3)   
        if return_latent:
            return [x, torch.clone(x)]
        else:
            return x


class GNO(torch.nn.Module):
    """current best"""
    def __init__(
            self,
            width: int,
            ker_width: int,
            depth: int,
            ker_in: int,            
            in_width: int = 1, # total number of node features: 3 coords x 10 frames + 4 embedding
            out_width: int = 1,
            num_embeddings: int = 20,
            embedding_dim: int = 4,
            x_position_dim: int = 3,
            num_nodes: int = 28,
            window_size: int = 10,
            num_layers: int = 10, #10, # 2,3,4,5,10 produce roughly same result; 1 is worse (for frame_step 100). For frame_step=1, num_layers=1 is best
            #mlp_num_layers: int = 2,
            #mlp_width: int = 128
    ) -> None:
        super(GNO, self).__init__()
        self.depth = depth
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.x_position_dim = x_position_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.window_size = window_size
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.fc1 = torch.nn.Linear(in_width, width)
        kernel = DenseNet([ker_in, ker_width, ker_width, width ** 2], torch.nn.ReLU, normalize=False)
        #kernel = DenseNet([ker_in, width ** 2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        #self.conv2 = NNConv_old(width, width, kernel, aggr="mean")
        #self.fcd = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, out_width)
        lstm_width = num_nodes * x_position_dim
        self.lstm = nn.LSTM(input_size=lstm_width, hidden_size=lstm_width, num_layers=num_layers)
        self.lstm_fc0 = torch.nn.Linear(lstm_width, lstm_width)
        self.lstm_fc1 = torch.nn.Linear(lstm_width * window_size, lstm_width)        
        #self.dropout = nn.Dropout(p=0.1)
        #self.fc_final = torch.nn.Linear(self.x_position_dim * 2, x_position_dim)
        #self.fc_final = DenseNet([self.x_position_dim * 2] + [mlp_width] * (mlp_num_layers - 1) + [x_position_dim], torch.nn.ReLU, out_nonlinearity=None, normalize=True)


    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        #x_position = self.dropout(data.x_position)
        x_position = data.x_position
        x1 = x_position
        x1 = x1.reshape(-1, self.window_size, x1.shape[-2] * x1.shape[-1])
        x1 = torch.swapaxes(x1, 0, 1)
        x1 = self.lstm_fc0(x1)
        self.lstm.flatten_parameters()
        hidden = (torch.zeros(self.num_layers, x1.shape[-2], x1.shape[-1]).cuda(),
                  torch.zeros(self.num_layers, x1.shape[-2], x1.shape[-1]).cuda())
        x1, hidden = self.lstm(x1, hidden)
        x1 = torch.swapaxes(x1, 0, 1).reshape(-1, self.num_nodes * self.x_position_dim * self.window_size)
        #x1 = self.dropout(x1)
        x1 = self.lstm_fc1(x1)
        #print('x1', x1.shape) # x1 torch.Size([256, 84])
        x1 = x1.reshape(-1, 3)   

        edge_index, edge_attr = data.edge_index, data.edge_attr
        x2 = x_position.reshape(-1, self.window_size, self.num_nodes, 3)
        x2 = torch.swapaxes(x2, 1, 2)
        emb = self.emb(data.x_aminoacid)
        x2 = x2.reshape(emb.shape[0], -1)
        x2 = torch.cat((emb, x2), dim=1)
        #x2 = self.dropout(x2)
        x2 = F.relu(self.fc1(x2))
        for k in range(self.depth):
            x2 = F.relu(self.conv1(x2, edge_index, edge_attr))
        #for k in range(self.depth):
        #    x = F.relu(self.conv2(x, edge_index, edge_attr))
        if return_latent:
            latent_dim = torch.clone(x)
        x2 = self.fc2(x2)
        #print('x2_', x2.shape) # x2_ torch.Size([7168, 3])
        #x2 = x2.reshape(-1, self.num_nodes * self.x_position_dim)
        #print('x2', x2.shape) # x2 torch.Size([256, 84])
        
        #x = torch.cat((x1, x2), dim=1)
        #x = self.dropout(x)
        #print('x', x.shape) #x torch.Size([256, 168])
        #x = self.fc_final(x)
        #print('xf', x.shape) #xf torch.Size([256, 84])
        x = (x1 + x2) / 2
        #x = x1 + x2
        x = x.reshape(-1, self.x_position_dim)
        #sys.exit(0)
        if return_latent:
            return [x, latent_dim]
        else:
            return x


# TODO: try reversing order: GNN followed by LSTM
class GNO_old(torch.nn.Module):
    def __init__(
            self,
            width: int,
            ker_width: int,
            depth: int,
            ker_in: int,            
            in_width: int = 1, # total number of node features: 3 coords x 10 frames + 4 embedding
            out_width: int = 1,
            num_embeddings: int = 20,
            embedding_dim: int = 4,
            x_position_dim: int = 3,
            num_nodes: int = 28,
            window_size: int = 10,
            num_layers: int = 10, # 2,3,4,5,10 produce roughly same result; 1 is worse (for frame_step 100). For frame_step=1, num_layers=1 is best
    ) -> None:
        super(GNO, self).__init__()
        self.depth = depth
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.x_position_dim = x_position_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.window_size = window_size
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.fc1 = torch.nn.Linear(in_width, width)
        kernel = DenseNet([ker_in, ker_width, ker_width, width ** 2], torch.nn.ReLU)
        #kernel = DenseNet([ker_in, width ** 2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        #self.conv2 = NNConv_old(width, width, kernel, aggr="mean")
        #self.fcd = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, out_width)

        lstm_width = num_nodes * x_position_dim
        self.lstm = nn.LSTM(input_size=lstm_width, hidden_size=lstm_width, num_layers=num_layers)
        self.lstm_fc0 = torch.nn.Linear(lstm_width, lstm_width)
        self.lstm_fc1 = torch.nn.Linear(lstm_width * window_size, lstm_width * window_size)        

    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        x = data.x_position
        #print('x0', x.shape)
        x = x.reshape(-1, self.window_size, x.shape[-2] * x.shape[-1])
        x = torch.swapaxes(x, 0, 1)
        x = self.lstm_fc0(x)
        self.lstm.flatten_parameters()
        hidden = (torch.zeros(self.num_layers, x.shape[-2], x.shape[-1]).cuda(),
                  torch.zeros(self.num_layers, x.shape[-2], x.shape[-1]).cuda())
        x, hidden = self.lstm(x, hidden)
        #print('x0', x.shape) # x0 torch.Size([10, 256, 84])
        x = self.lstm_fc1(torch.swapaxes(x, 0, 1).reshape(-1, self.num_nodes * self.x_position_dim * self.window_size))
        #print('x1', x.shape) # x1 torch.Size([256, 840])
        x = x.reshape(x.shape[0], self.window_size, self.num_nodes, self.x_position_dim)
        #x = torch.swapaxes(x, 0, 1)
        #print('x1b', x.shape) # x1b torch.Size([256, 10, 28, 3])
        #x = x.reshape(x.shape[0], x.shape[1], self.num_nodes, self.x_position_dim)
        #print('x1c', x.shape)
        x = torch.swapaxes(x, 1, 2)
        #print('x1d', x.shape) # x1d torch.Size([256, 28, 10, 3])
        edge_index, edge_attr = data.edge_index, data.edge_attr
        #x = data.x_position.reshape(-1, self.window_size, self.num_nodes, 3)
        #x = torch.swapaxes(x, 1, 2)
        emb = self.emb(data.x_aminoacid)
        x = x.reshape(emb.shape[0], -1)
        #print('x2', x.shape) # x2 torch.Size([7168, 30])
        x = torch.cat((emb, x), dim=1)
        #print('x3', x.shape)
        x = F.relu(self.fc1(x))
        for k in range(self.depth):
            #x = self.fcd(F.relu(self.conv1(x, edge_index, edge_attr)))
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        #for k in range(self.depth):
        #    x = F.relu(self.conv2(x, edge_index, edge_attr))
        if return_latent:
            latent_dim = torch.clone(x)
        x = self.fc2(x)
        #sys.exit(0)
        if return_latent:
            return [x, latent_dim]
        else:
            return x


class GNO_v2(torch.nn.Module):
    def __init__(
            self,
            width: int,
            ker_width: int,
            depth: int,
            ker_in: int,            
            in_width: int = 1, # total number of node features: 3 coords x 10 frames + 4 embedding
            out_width: int = 1,
            num_embeddings: int = 20,
            embedding_dim: int = 4,
            x_position_dim: int = 3,
            num_nodes: int = 28,
            window_size: int = 10,
            num_layers: int = 10, # 2,3,4,5,10 produce roughly same result; 1 is worse (for frame_step 100). For frame_step=1, num_layers=1 is best
    ) -> None:
        super(GNO, self).__init__()
        self.depth = depth
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.x_position_dim = x_position_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.window_size = window_size
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.fc1 = torch.nn.Linear(in_width, width)
        kernel = DenseNet([ker_in, ker_width, ker_width, width ** 2], torch.nn.ReLU)
        #kernel = DenseNet([ker_in, width ** 2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        #self.conv2 = NNConv_old(width, width, kernel, aggr="mean")
        #self.fcd = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, self.window_size * self.x_position_dim) #out_width)

        lstm_width = num_nodes * x_position_dim
        self.lstm = nn.LSTM(input_size=lstm_width, hidden_size=lstm_width, num_layers=num_layers)
        self.lstm_fc0 = torch.nn.Linear(lstm_width, lstm_width)
        self.lstm_fc1 = torch.nn.Linear(lstm_width * window_size, lstm_width)        

    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        x = data.x_position

        # GNN first
        #print('x0', x.shape) #torch.Size([2560, 28, 3])
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = data.x_position.reshape(-1, self.window_size, self.num_nodes, 3)
        #print('x1', x.shape) # x1 torch.Size([256, 10, 28, 3])
        x = torch.swapaxes(x, 1, 2)
        emb = self.emb(data.x_aminoacid)
        x = x.reshape(emb.shape[0], -1)
        #print('x2', x.shape) # x2 torch.Size([7168, 30])
        x = torch.cat((emb, x), dim=1)
        #print('x3', x.shape) # x3 torch.Size([7168, 34])
        x = F.relu(self.fc1(x)) # lifts 34 to 128
        for k in range(self.depth):
            #x = self.fcd(F.relu(self.conv1(x, edge_index, edge_attr)))
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        #for k in range(self.depth):
        #    x = F.relu(self.conv2(x, edge_index, edge_attr))
        #print('x4', x.shape) # x4 torch.Size([7168, 128])
        x = self.fc2(x)
        #print('x5', x.shape) # x5 torch.Size([7168, 30])

        # Then LSTM
        x = x.reshape(-1, self.num_nodes, self.window_size, self.x_position_dim)
        #print('x5b', x.shape) #x5b torch.Size([256, 28, 10, 3])
        x = torch.swapaxes(x, 1, 2)
        #print('x5c', x.shape) #x5c torch.Size([256, 10, 28, 3])
        x = x.reshape(-1, self.window_size, self.num_nodes * self.x_position_dim)
        #print('x5d', x.shape) #x5d torch.Size([256, 10, 84])
        x = torch.swapaxes(x, 0, 1)
        #print('x5e', x.shape) #x5e torch.Size([10, 256, 84])
        x = self.lstm_fc0(x)
        self.lstm.flatten_parameters()
        hidden = (torch.zeros(self.num_layers, x.shape[-2], x.shape[-1]).cuda(),
                  torch.zeros(self.num_layers, x.shape[-2], x.shape[-1]).cuda())
        x, hidden = self.lstm(x, hidden)
        #print('x5f', x.shape) # x5f torch.Size([10, 256, 84])
        x = torch.swapaxes(x, 0, 1).reshape(-1, self.num_nodes * self.x_position_dim * self.window_size)
        if return_latent:
            latent_dim = torch.clone(x)
        x = self.lstm_fc1(x)
        #print('x6', x.shape) #  torch.Size([256, 84])
        x = x.reshape(-1, 3)   
        if return_latent:
            return [x, latent_dim]
        else:
            return x

class BasicNN(torch.nn.Module):
    """
    Linear transformation on all coordinates; no non-linearity!
    """
    def __init__(
            self,
            in_width: int = 1,
            out_width: int = 1,
    ) -> None:
        super(BasicNN, self).__init__()
        self.in_width = in_width
        self.out_width = out_width
        self.fc1 = torch.nn.Linear(in_width, out_width)

    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        x = data.x_position.reshape(-1, self.in_width) #args.window_size * self.num_nodes * 3
        x = self.fc1(x)
        x = x.reshape(-1, 3)
        #print("x:", x.shape)
        #sys.exit(0)        
        if return_latent:
            return [x, torch.clone(x)]
        else:
            return x


class BasicMLP(torch.nn.Module):
    """
    Linear transformation on all coordinates, followed by one non-linear activation function
    """
    def __init__(
            self,
            in_width: int = 1,
            out_width: int = 1,
            n_layers: int = 2,
            normalize: bool = True,
            num_embeddings: int = 20,
            embedding_dim: int = 4, #4, # no gain in providing AA information!
            num_nodes: int = 28
            #n_layers: int = 1,
            #num_nodes: int = 28,
            #window_size: int = 1
    ) -> None:
        super(BasicMLP, self).__init__()
        self.in_width = in_width
        self.out_width = out_width
        self.kernel = DenseNet([in_width + embedding_dim * num_nodes] * n_layers + [out_width], torch.nn.ReLU, out_nonlinearity=None, normalize=normalize)
        if embedding_dim > 0:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes


    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        #print(data.x_position.shape)
        x = data.x_position #.reshape(-1, self.in_width) #args.window_size * self.num_nodes * 3
        x = x.reshape(-1, self.in_width)
        #print('x0', x.shape)
        if self.embedding_dim > 0:
            emb = self.emb(data.x_aminoacid).reshape(-1, self.num_nodes * self.embedding_dim)
            #print('emb', emb.shape)
            x = x.reshape(emb.shape[0], -1)
            #print('x1', x.shape)
            x = torch.cat((emb, x), dim=1)
            #print('xemb', x.shape)
            
        x = self.kernel(x)
        #print("x1", x.shape)
        x = x.reshape(-1, 3)
        #print("xf", x.shape)
        #sys.exit(0)        
        if return_latent:
            return [x, torch.clone(x)]
        else:
            return x



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--run_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler_step", type=int, default=50)
    parser.add_argument("--scheduler_gamma", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--out_width", type=int, default=3)
    parser.add_argument("--kernel_width", type=int, default=1024)
    parser.add_argument("--GNN_depth", type=int, default=6)
    #parser.add_argument("--node_features", type=int, default=7)
    parser.add_argument("--edge_features", type=int, default=6)
    parser.add_argument("--num_embeddings", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=4)
    parser.add_argument("--split_pct", type=float, default=0.8)
    parser.add_argument("--num_data_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", type=str, default="False")
    parser.add_argument("--non_blocking", type=str, default="False")
    parser.add_argument("--num_movie_frames", type=int, default=5)
    parser.add_argument("--plot_per_epochs", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=10, help="Size of window to feed into network")

    #parser.add_argument("--plot_latent", type=bool, default=True)
    #parser.add_argument("--generate_movie", type=bool, default=True)
    parser.set_defaults(plot_latent=True)
    parser.set_defaults(generate_movie=True)
    parser.add_argument('--dont-plot_latent', dest='plot_latent', action='store_false')
    parser.add_argument('--dont-generate_movie', dest='generate_movie', action='store_false')
    parser.add_argument("--augment_by_reversing_prob", type=float, default=0)
    parser.add_argument("--augment_by_rotating180_prob", type=float, default=0) # buggy!
    parser.add_argument("--augment_by_translating_prob", type=float, default=0)
    parser.add_argument("--augment_with_noise_mu", type=float, default=0) 
    parser.add_argument('--n_frames', type=int, default=250000)
    parser.add_argument('--model', type=str, default='KernelNN', choices=['KernelNN', 'BasicNN', 'BasicLSTM', 'BasicMLP', 'BasicGNN', 'KernelNNFixed', 'GNO'])
    parser.add_argument('--frame_step', type=int, default=1, help="Number of frames in the raw trajectory between frames chosen for the window, i.e., using window_size frames every frame_step frames, predict the frame that comes after frame_step frames")
    parser.add_argument("--MLP_n_layers", type=int, default=2, help="Number of layers in the BasicMLP")
    parser.add_argument("--LSTM_n_layers", type=int, default=10, help="Number of layers in the BasicLSTM")

    args = parser.parse_args()

    # Validation of arguments
    if not args.data_path.exists():
        raise ValueError(f"data_path does not exist: {args.data_path}")
    args.persistent_workers = args.persistent_workers == "True"
    args.non_blocking = args.non_blocking == "True"

    # use the weights and biases trial name to store output
    if wandb.run.name:
        args.run_path = args.run_path / wandb.run.name
    # Make output directory
    if not os.path.isdir(args.run_path):
        args.run_path.mkdir()
    else:
        print(f'{args.run_path} exists, over-writing...')
    return args


def construct_pairdata(x_position, x_aminoacid, threshold: float = 8.0) -> PairData:
    contact_map = (distance_matrix(x_position[-1], x_position[-1]) < threshold).astype("int8")
    sparse_contact_map = coo_matrix(contact_map)
    # print(sparse_contact_map.row)
    # print(sparse_contact_map.col)
    # Get adjacency list
    edge_index = np.array([sparse_contact_map.row, sparse_contact_map.col])
    # Get edge attributes with shape (num_edges, num_edge_features)
    # Each edge attribute is the positions of both atoms A,B
    # And looks like [Ax, Ay, Az, Bx, By, Bz]
    edge_attr = np.array(
        [
            np.concatenate(
                (x_position[-1, i, :], x_position[-1, j, :])
            ).flatten()
            for i, j in zip(edge_index[0], edge_index[1])
        ]
    )

    x_position = torch.from_numpy(x_position).to(torch.float32)
    edge_index = torch.from_numpy(edge_index).to(torch.long)
    edge_attr = torch.from_numpy(edge_attr).to(torch.float32)

    # Construct torch_geometric data object
    data = PairData(
        x_aminoacid=x_aminoacid,
        x_position=x_position,
        edge_attr=edge_attr,
        edge_index=edge_index,
    )

    return data


def recursive_propagation(model, dataset, device, num_steps: int, starting_points: list, threshold: float = 8.0):
    forecasts = []
    model.eval()
    with torch.no_grad():
        for start in starting_points:
            input_ = dataset[start].to(device)
            for i in range(start, start+num_steps):
                input_ = input_.to(device)
                output = model.module(input_, single_example=True)
                # generate new x positions
                last_window = input_.x_position.cpu().numpy()[1:, :, :]
                out_x_position = output.detach().cpu().numpy()
                out_x_position = np.expand_dims(out_x_position, 0)
                new_x_position = np.vstack([last_window, out_x_position])
                input_ = construct_pairdata(new_x_position, input_.x_aminoacid, threshold=threshold)
                forecasts.append(input_.to("cpu"))

    return forecasts


def get_contact_map(pair_data):
    row = pair_data.edge_index.cpu().numpy()[0]
    col = pair_data.edge_index.cpu().numpy()[1]
    val = np.ones(len(row))
    dense_contact_map = coo_matrix((val, (row, col)), shape=(28, 28)).toarray()
    return dense_contact_map


def make_propagation_movie(model, dataset, device, num_steps=5, starting_points=[0, 25, 50]):
    forecast = recursive_propagation(model, dataset, device, num_steps=num_steps, starting_points=starting_points)
    filenames = []
    for starting_point in starting_points:
        for i in range(starting_point, starting_point+num_steps):
            forecast_cm = get_contact_map(forecast.pop(0))
            real_cm = get_contact_map(dataset[i + 1])
            fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
            ax[0].imshow(forecast_cm, cmap="cividis")
            ax[1].imshow(real_cm, cmap="cividis")
            fig.suptitle("Time Step {}".format(i + 1))
            ax[0].set_title("Forecast")
            ax[1].set_title("Real")
            filename = '/tmp/gno_movie/frame{}.png'.format(i + 1)
            filenames.append(filename)
            plt.savefig(filename, dpi=150)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('/tmp/gno_movie/movie.mp4', images)

def train(
    model, train_loader, optimizer, loss_fn, device,
):
    model.train()
    avg_loss = 0.0
    avg_mse = 0.0
    mse_fn = torch.nn.MSELoss()
    for batch in tqdm(train_loader):
        # batch = batch.to(device, non_blocking=args.non_blocking)
        # TODO: normally augmentations would go here, after a normal batch is produced
        optimizer.zero_grad()
        out = model(batch)
        # mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1, 1))
        # mse.backward()
        # loss = torch.norm(out.view(-1) - batch.y.view(-1), 1)
        # loss.backward()
        concat_y = torch.cat([data.y for data in batch]).to(out.device)
        l2 = loss_fn(out.view(args.batch_size, -1), concat_y.view(args.batch_size, -1))
        l2.backward()
        mse_loss = mse_fn(out, concat_y)
        optimizer.step()
        avg_loss += l2.item()
        avg_mse += mse_loss.item()

    avg_loss /= len(train_loader)
    avg_mse /= len(train_loader)

    return avg_loss, avg_mse


def validate(model, valid_loader, loss_fn, device):
    model.eval()
    avg_loss = 0.0
    avg_mse = 0.0
    mse_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in valid_loader:
            # data = batch.to(device, non_blocking=args.non_blocking)
            out = model(batch)
            concat_y = torch.cat([data.y for data in batch]).to(out.device)
            avg_loss += loss_fn(
                out.view(args.batch_size, -1), concat_y.view(args.batch_size, -1)
            ).item() # average loss per batch -- still a function of batch size
            avg_mse += mse_fn(out, concat_y)
    avg_loss /= len(valid_loader)
    avg_mse /= len(valid_loader)
    return avg_loss, avg_mse


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set available number of cores
    torch.set_num_threads(1 if args.num_data_workers == 0 else args.num_data_workers)

    # Setup training and validation datasets
    # dataset = ContactMapDataset(args.data_path, window_size=args.window_size, n_frames=args.n_frames, frame_step=args.frame_step)
    # train_loader, valid_loader, train_dataset, valid_dataset = train_valid_split(
    #     dataset,
    #     args.split_pct,
    #     method="partition",
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     pin_memory=True,
    #     num_workers=args.num_data_workers,
    #     prefetch_factor=args.prefetch_factor,
    #     persistent_workers=args.persistent_workers,
    #     augment_by_reversing_prob=args.augment_by_reversing_prob,
    #     augment_by_rotating180_prob=args.augment_by_rotating180_prob,
    #     augment_by_translating_prob=args.augment_by_translating_prob,
    #     augment_with_noise_mu=args.augment_with_noise_mu
    # )

    train_length = int(args.n_frames * args.split_pct)
    valid_length = args.n_frames - train_length

    train_dataset = ContactMapNewDataset(
        args.data_path, window_size=args.window_size, frames_range=[0, train_length], frame_step=args.frame_step,
        augment_by_reversing_prob=args.augment_by_reversing_prob,
        augment_by_rotating180_prob=args.augment_by_rotating180_prob,
        augment_by_translating_prob=args.augment_by_translating_prob,
        augment_with_noise_mu=args.augment_with_noise_mu
    )
    print(train_dataset)
    valid_dataset = ContactMapNewDataset(
        args.data_path, window_size=args.window_size, frames_range=[train_length, args.n_frames], frame_step=args.frame_step,
        augment_by_reversing_prob=0,
        augment_by_rotating180_prob=0,
        augment_by_translating_prob=0,
        augment_with_noise_mu=0
    )
    print(valid_dataset)
    print("Created dataset")
    train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.num_data_workers, prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers)
    valid_loader = DataListLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.num_data_workers, prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers)

    print("Split training and validation sets")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model, optimizer, loss function and scheduler
    if args.model == 'KernelNN':
        node_features = args.out_width + args.embedding_dim
        model = DataParallel(KernelNN(
            args.width,
            args.kernel_width,
            args.GNN_depth,
            args.edge_features,
            node_features,
            args.out_width,
            args.num_embeddings,
            args.embedding_dim,
            num_nodes=train_dataset.num_nodes,
            window_size=args.window_size
        )).to(device)
    elif args.model == 'KernelNNFixed':
        node_features = args.out_width + args.embedding_dim
        model = DataParallel(KernelNNFixed(
            args.width,
            args.kernel_width,
            args.GNN_depth,
            args.edge_features,
            node_features,
            args.out_width,
            args.num_embeddings,
            args.embedding_dim,
            num_nodes=train_dataset.num_nodes,
            window_size=args.window_size
        )).to(device)
    elif args.model == 'BasicGNN':
        node_features = args.out_width * args.window_size + args.embedding_dim
        model = DataParallel(BasicGNN(
            args.width,
            args.kernel_width,
            args.GNN_depth,
            args.edge_features,
            node_features,
            args.out_width,
            args.num_embeddings,
            args.embedding_dim,
            num_nodes=train_dataset.num_nodes,
            window_size=args.window_size
        )).to(device)    
    elif args.model == 'GNO':
        node_features = args.out_width * args.window_size + args.embedding_dim
        #node_features = args.out_width + args.embedding_dim
        model = DataParallel(GNO(
            args.width,
            args.kernel_width,
            args.GNN_depth,
            args.edge_features,
            node_features,
            args.out_width,
            args.num_embeddings,
            args.embedding_dim,
            num_nodes=train_dataset.num_nodes,
            window_size=args.window_size,
            num_layers=args.LSTM_n_layers,
        )).to(device)    
    elif args.model == 'BasicLSTM':
        model = DataParallel(BasicLSTM(
            train_dataset.num_nodes * 3,
            num_layers=args.LSTM_n_layers,
            window_size=args.window_size
        )).to(device)        
    elif args.model == 'BasicMLP':
        model = DataParallel(BasicMLP(
            args.window_size * train_dataset.num_nodes * 3,
            train_dataset.num_nodes * args.out_width,
            n_layers = args.MLP_n_layers
        )).to(device)        
    elif args.model == 'BasicNN':
        model = DataParallel(BasicNN(
            args.window_size * train_dataset.num_nodes * 3,
            train_dataset.num_nodes * args.out_width
        )).to(device)
    else:
        raise RuntimeError(f'Model {args.model} not supported')

    print("Initialized model", model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    loss_fn = LpLoss(size_average=False) # set size_average=True to get loss per example

    print("Started training")

    # Start training
    best_loss = float("inf")
    for epoch in range(args.epochs):
        time = default_timer()
        avg_train_loss, avg_train_mse = train(model, train_loader, optimizer, loss_fn, device)
        avg_valid_loss, avg_valid_mse = validate(model, valid_loader, loss_fn, device)
        video = None
        if args.generate_movie and (epoch % args.plot_per_epochs == 0):
            make_propagation_movie(model, valid_dataset, device, args.num_movie_frames)
            video = wandb.Video('/tmp/gno_movie/movie.mp4', fps=2, format="mp4")
        if args.plot_latent and (epoch % args.plot_per_epochs == 0):
            with torch.no_grad():
                latent_spaces = []
                for inference_step in range(10000):

                    out, latent = model.module.forward(train_dataset[inference_step+133000].cuda(), return_latent=True, single_example=True)
                    latent = latent.cpu().numpy().flatten()
                    latent_spaces.append(latent)

                latent_spaces = np.array(latent_spaces)
                color_dict = {'RMSD': train_dataset.rmsd_values[133000:143000]}
                out_html = log_latent_visualization(latent_spaces, color_dict, '/tmp/latent_html/', epoch=epoch, method="PCA")
                html_plot = wandb.Html(out_html['RMSD'], inject=False)
        else:
            html_plot = None
        wandb.log({'avg_train_loss': avg_train_loss, 'avg_valid_loss': avg_valid_loss,
                   'avg_train_mse': avg_train_mse, 'avg_valid_mse': avg_valid_mse,
                   'valid_prediction_video': video, 'RMSD_latent_plot': html_plot})
        scheduler.step()
        print(
            f"Epoch: {epoch}"
            f"\tTime: {default_timer() - time}"
            f"\ttrain_loss: {avg_train_loss}"
            f"\tvalid_loss: {avg_valid_loss}"
            f"\ntrain_dataset: {train_dataset.n_aug}"
            f"\nvalid_dataset: {valid_dataset.n_aug}"
        )
        sys.stdout.flush()
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
    wandb.init(project="bba_gno")
    args = parse_args()
    print(args)
    wandb.config.update(args)
    main()

# TODO:
# add batchnorm to kernel
# add all edges to GNN
# augmentation
# GeLU