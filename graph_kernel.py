import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, Union
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

from dataset import ContactMapDataset, PairData

from mdlearn.utils import log_latent_visualization

EPS = 1e-15

PathLike = Union[str, Path]


def train_valid_split(
        dataset: Dataset, split_pct: float = 0.8, method: str = "random", **kwargs
) -> Tuple[DataListLoader, DataListLoader]:
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
        pdb.set_trace()
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
            x_position_dim: int = 3
    ) -> None:
        super(KernelNN, self).__init__()
        self.depth = depth
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.x_position_dim = x_position_dim

        self.lstm = nn.LSTM(x_position_dim, x_position_dim)
        self.lstm_fc = torch.nn.Linear(x_position_dim, x_position_dim+embedding_dim)

        # self.emb = nn.Embedding(num_embeddings, embedding_dim)

        # self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width ** 2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        self.conv2 = NNConv_old(width, width, kernel, aggr="mean")

        self.fc2 = torch.nn.Linear(width, out_width)

    def forward(self, data: PairData, return_latent: bool = False, single_example: bool = False) -> [torch.Tensor, Optional[torch.tensor]]:
        pdb.set_trace()
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = data.x_position.reshape(-1, args.window_size, args.num_residues, 3)
        x = torch.swapaxes(x, 0, 1)
        hidden = (torch.zeros(1, args.num_residues, 3).cuda(),
                  torch.zeros(1, args.num_residues, 3).cuda())
        for i in x:
            x, hidden = self.lstm(i, hidden)
        # x, hidden = self.lstm(x)
        # take the last time slice, we don't want all of them
        # x = x[-args.batch_size:]
        x = F.relu(self.lstm_fc(x))
        # Use an embedding layer to map the onehot aminoacid vector to
        # a dense vector and then concatenate the result with the positions
        # emb = self.emb(data.x_aminoacid.view(args.batch_size, -1, self.num_embeddings))
        # emb = self.emb(data.x_aminoacid)
        # x = x.reshape(emb.shape[0], -1)
        # print("data.x_aminoacid", data.x_aminoacid.shape)
        # print("data.x_position:", data.x_position.shape)
        # x = torch.cat((emb, x), dim=1)
        # print("x:", x.shape)
        # x = F.relu(self.fc1(x))
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        for k in range(self.depth):
            x = F.relu(self.conv2(x, edge_index, edge_attr))
        if return_latent:
            latent_dim = torch.clone(x)
        x = self.fc2(x)
        if return_latent:
            return [x, latent_dim]
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
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--node_features", type=int, default=7)
    parser.add_argument("--edge_features", type=int, default=6)
    parser.add_argument("--num_embeddings", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=4)
    parser.add_argument("--split_pct", type=float, default=0.8)
    parser.add_argument("--num_data_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", type=str, default="False")
    parser.add_argument("--non_blocking", type=str, default="False")
    parser.add_argument("--generate_movie", type=bool, default=True)
    parser.add_argument("--num_movie_frames", type=int, default=5)
    parser.add_argument("--plot_latent", type=bool, default=True)
    parser.add_argument("--plot_per_epochs", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=10, help="Size of window to feed into network")
    parser.add_argument("--num_residues", type=int, default=28)
    parser.add_argument("--latent_space_starting_frame", type=int, default=133000)
    parser.add_argument("--latent_space_num_frames", type=int, default=10000)
    parser.add_argument("--node_features_path", type=Path, default=None)

    args = parser.parse_args()

    # Validation of arguments
    if not args.data_path.exists():
        raise ValueError(f"data_path does not exist: {args.data_path}")
    args.persistent_workers = args.persistent_workers == "True"
    args.non_blocking = args.non_blocking == "True"

    # use the weights and biases trial name to store output
    args.run_path = args.run_path / wandb.run.name
    # Make output directory
    args.run_path.mkdir()

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
        # x_aminoacid=x_aminoacid,
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
    dense_contact_map = coo_matrix((val, (row, col)), shape=(args.num_residues, args.num_residues)).toarray()
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

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    avg_loss = 0.0
    avg_mse = 0.0
    mse_fn = torch.nn.MSELoss()
    for batch in tqdm(train_loader):
        # batch = batch.to(device, non_blocking=args.non_blocking)

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
            ).item()
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
    dataset = ContactMapDataset(args.data_path, window_size=args.window_size,
                                node_feature_dset_path=args.node_features_path)

    print("Created dataset")

    train_loader, valid_loader, train_dataset, valid_dataset = train_valid_split(
        dataset,
        args.split_pct,
        method="partition",
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=args.num_data_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    print("Split training and validation sets")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model, optimizer, loss function and scheduler
    model = DataParallel(KernelNN(
        args.width,
        args.kernel_width,
        args.depth,
        args.edge_features,
        args.node_features,
        args.out_width,
        args.num_embeddings,
        args.embedding_dim,
    )).to(device)

    print("Initialized model")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    loss_fn = LpLoss(size_average=False)

    print("Started training")

    # calculate the starting points for the prediction propagation movie
    if args.generate_movie:
        total_steps = len(valid_dataset) - 10
        potential_starts = list(range(0, total_steps, args.window_size))
        if len(potential_starts) < 3:
            starting_points = potential_starts
        else:
            starting_points = []
            # first window
            starting_points.append(0)
            # middle window
            starting_points.append(potential_starts[(len(potential_starts)//2)])
            # last window
            starting_points.append(potential_starts[-1])

    pdb.set_trace()

    # Start training
    best_loss = float("inf")
    # save rmsd paints
    # np.save(args.run_path / 'rmsd.npy', valid_dataset.rmsd_values[
    #                                   args.latent_space_starting_frame:args.latent_space_starting_frame + args.latent_space_num_frames])

    for epoch in range(args.epochs):
        time = default_timer()
        avg_train_loss, avg_train_mse = train(model, train_loader, optimizer, loss_fn, device)
        avg_valid_loss, avg_valid_mse = validate(model, valid_loader, loss_fn, device)
        video = None
        if args.generate_movie and (epoch % args.plot_per_epochs == 0):
            make_propagation_movie(model, valid_dataset, device, args.num_movie_frames, starting_points=starting_points)
            video = wandb.Video('/tmp/gno_movie/movie.mp4', fps=2, format="mp4")
        if args.plot_latent and (epoch % args.plot_per_epochs == 0):
            with torch.no_grad():
                latent_spaces = []
                for inference_step in range(args.latent_space_num_frames):

                    out, latent = model.module.forward(dataset[inference_step+args.latent_space_starting_frame].cuda(), return_latent=True, single_example=True)
                    latent = latent.cpu().numpy().flatten()
                    latent_spaces.append(latent)

                latent_spaces = np.array(latent_spaces)
                # save in directory
                np.save(args.run_path/'latent_epoch{}.npy'.format(epoch), latent_spaces)
                color_dict = {'RMSD': valid_dataset.rmsd_values[
                                      args.latent_space_starting_frame:args.latent_space_starting_frame + args.latent_space_num_frames]}

                print(len(color_dict['RMSD']))
                print(len(latent_spaces))
                out_html = log_latent_visualization(latent_spaces, color_dict, '/tmp/latent_html/', epoch=epoch, method="PCA")
                html_plot = wandb.Html(out_html['RMSD'], inject=False)
                out_html = log_latent_visualization(latent_spaces, color_dict, '/tmp/latent_html/', epoch=epoch,
                                                    method="TSNE")
                html_plot2 = wandb.Html(out_html['RMSD'], inject=False)
        else:
            html_plot = None
            html_plot2 = None
        wandb.log({'avg_train_loss': avg_train_loss, 'avg_valid_loss': avg_valid_loss,
                   'avg_train_mse': avg_train_mse, 'avg_valid_mse': avg_valid_mse,
                   'valid_prediction_video': video, 'PCA_RMSD_latent_plot': html_plot,
                   'TSNE_RMSD_latent_plot': html_plot2})
        scheduler.step()
        print(
            f"Epoch: {epoch}"
            f"\tTime: {default_timer() - time}"
            f"\ttrain_loss: {avg_train_loss}"
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
    wandb.init(project="bba_gno")
    args = parse_args()
    wandb.config.update(args)
    main()
