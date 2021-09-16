import torch
import h5py
import numpy as np
from pathlib import Path
from typing import Union, Optional
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor


PathLike = Union[str, Path]


def aminoacid_int_to_onehot(labels):
    # 20 amino acids total
    onehot = np.zeros((len(labels), 20))
    for i, label in enumerate(labels):
        # labels are 1 index ranging from [1, 20]
        onehot[i][label - 1] = 1
    return onehot


class PairData(Data):
    def __init__(
        self,
        x: OptTensor = None,
        y: OptTensor = None,
        edge_attr: OptTensor = None,
        edge_index_s: OptTensor = None,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.edge_attr = edge_attr
        self.edge_index_s = edge_index_s

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x.size(0)
        if key == "edge_index_t":
            return self.x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        self.edge_attr = self.edge_attr.pin_memory()
        self.edge_index_s = self.edge_index_s.pin_memory()
        return self


class ContactMapDataset(Dataset):
    """
    PyTorch Dataset class to load contact matrix data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """

    def __init__(
        self,
        path: PathLike,
        edge_index_dset_name: str = "contact_map",
        edge_attr_dset_name: str = "point_cloud",
        node_feature_dset_name: Optional[str] = "amino_acids",
        node_feature: str = "amino_acid_onehot",
        constant_num_node_features: int = 20,
        window_size: int = 1,
        horizon: int = 1,
    ):
        """
        Parameters
        ----------
        path : PathLike
            Path to h5 file containing contact matrices.
        edge_index_dset_name : str
            Name of contact map dataset in HDF5 file.
        edge_attr_dset_name : str
            Name of positions dataset in HDF5 file.
        node_feature : str
            Type of node features to use. Available options are `constant`,
            `identity`, and `amino_acid_onehot`. If `constant` is selected,
            `constant_num_node_features` must be selected.
        constant_num_node_features : int
            Number of node features when using constant `node_feature` vectors.
        window_size : int, default=1
            Number of timesteps considered for prediction.
        horizon : int, default=1
            How many time steps to predict ahead.

        Raises
        ------
        ValueError
            If :obj:`window_size` is greater than 1.
        ValueError
            If the sum of :obj:`window_size` and :obj:`horizon` is longer
            than the input data.
        """

        if window_size > 1:
            raise ValueError("Currently supports a max window_size of 1")

        self._constant_num_node_features = constant_num_node_features
        self.window_size = window_size
        self.horizon = horizon

        # Truncate dataset for shorter training time
        ntrain = 100000

        with h5py.File(path, "r", libver="latest", swmr=False) as f:
            # COO formated ragged arrays
            self.edge_indices = np.array(f[edge_index_dset_name][:ntrain])
            self.edge_attrs = np.array(f[edge_attr_dset_name][:ntrain])
            if node_feature_dset_name is not None:
                self._node_features_dset = f[node_feature_dset_name][...]

        if len(self.edge_indices) - self.window_size - self.horizon + 1 < 0:
            raise ValueError(
                "The sum of window_size and horizon is longer than the input data"
            )

        self.node_features = self._compute_node_features(node_feature)

    def _compute_node_features(self, node_feature: str) -> np.ndarray:
        if node_feature == "constant":
            node_features = np.ones((self.num_nodes, self._constant_num_node_features))
        elif node_feature == "identity":
            node_features = np.eye(self.num_nodes)
        elif node_feature == "amino_acid_onehot":
            node_features = aminoacid_int_to_onehot(self._node_features_dset)
        else:
            raise ValueError(f"node_feature: {node_feature} not supported.")
        return node_features

    def __len__(self):
        return len(self.edge_indices) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):

        pred_idx = idx + self.window_size + self.horizon - 1

        # Get node features (the node features are constant per-graph)
        node_features = self.node_features

        # Get adjacency list
        edge_index_s = self.edge_indices[idx].reshape(2, -1)  # [2, num_edges]

        # Get edge attributes with shape (num_edges, num_edge_features)
        # Each edge attribute is the positions of both atoms A,B
        # And looks like [Ax, Ay, Az, Bx, By, Bz]
        edge_attr = np.array(
            [
                np.concatenate(
                    (self.edge_attrs[idx, :, i], self.edge_attrs[idx, :, j])
                ).flatten()
                for i, j in zip(edge_index_s[0], edge_index_s[1])
            ]
        )

        # Get the raw xyz positions (num_nodes, 3) at the prediction index
        y = self.edge_attrs[pred_idx].transpose()

        # Convert to torch.Tensor
        node_features = torch.from_numpy(node_features).to(torch.float32)
        edge_index_s = torch.from_numpy(edge_index_s).to(torch.long)
        edge_attr = torch.from_numpy(edge_attr).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        # print("node_features:", node_features.shape)
        # print("edge_index_s:", edge_index_s.shape)
        # print("edge_attr:", edge_attr.shape)
        # print("y:", y.shape)

        # Construct torch_geometric data object
        data = PairData(
            x=node_features,
            y=y,
            edge_attr=edge_attr,
            edge_index_s=edge_index_s,
        )

        return data
