import torch
import h5py
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
import pdb
import random

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
        x_aminoacid: OptTensor = None,
        x_position: OptTensor = None,
        y: OptTensor = None,
        edge_attr: OptTensor = None,
        edge_index: OptTensor = None,
    ) -> None:
        super().__init__()
        self.x_aminoacid = x_aminoacid
        self.x_position = x_position
        self.y = y
        self.edge_attr = edge_attr
        self.edge_index = edge_index

    @property
    def num_nodes(self) -> int:
        return self.x_aminoacid.size(0)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def pin_memory(self):
        self.x_aminoacid = self.x_aminoacid.pin_memory()
        self.x_position = self.x_position.pin_memory()
        self.y = self.y.pin_memory()
        self.edge_attr = self.edge_attr.pin_memory()
        self.edge_index = self.edge_index.pin_memory()
        return self

    # TODO: implement augmentations here


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
        n_frames: int = 250000
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
        n_frames: int, default=250000
            Maximum number of trajectory frames to read
        Raises
        ------
        ValueError
            If :obj:`window_size` is greater than 1.
        ValueError
            If the sum of :obj:`window_size` and :obj:`horizon` is longer
            than the input data.
        """

        self._constant_num_node_features = constant_num_node_features
        self.window_size = window_size
        self.horizon = horizon

        with h5py.File(path, "r", libver="latest", swmr=False) as f:
            # COO formated ragged arrays
            self.edge_indices = np.array(f[edge_index_dset_name][:n_frames])
            self.edge_attrs = np.array(f[edge_attr_dset_name][:n_frames])
            # get the rmsd nums
            try:
                self.rmsd_values = np.array(f['rmsd'][:n_frames])
            except ValueError as e:
                print("Not able to load rmsd values...")
                self.rmsd_values = []
            if node_feature_dset_name is not None:
                self._node_features_dset = f[node_feature_dset_name][...]

        if len(self.edge_indices) - self.window_size - self.horizon + 1 < 0:
            raise ValueError(
                "The sum of window_size and horizon is longer than the input data"
            )

        # Put positions in order (N, num_nodes, 3)
        self.edge_attrs = np.transpose(self.edge_attrs, [0, 2, 1])
        #print(self._node_features_dset)
        #print(self._node_features_dset.shape)
        self.x_aminoacid = self._node_features_dset
        #self.x_aminoacid = self._compute_node_features(node_feature)
        self.x_aminoacid = torch.from_numpy(self.x_aminoacid).to(torch.long)

    @property
    def num_nodes(self) -> int:
        return self.x_aminoacid.size(0)

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

        # Get the positions (num_nodes, 3)
        x_position = self.edge_attrs[idx:idx+self.window_size]
        # x_position = self.edge_attrs[idx]

        # Get adjacency list
        edge_index = self.edge_indices[idx].reshape(2, -1)  # [2, num_edges]

        # Get edge attributes with shape (num_edges, num_edge_features)
        # Each edge attribute is the positions of both atoms A,B
        # And looks like [Ax, Ay, Az, Bx, By, Bz]
        edge_attr = np.array(
            [
                np.concatenate(
                    (self.edge_attrs[idx, i, :], self.edge_attrs[idx, j, :])
                ).flatten()
                for i, j in zip(edge_index[0], edge_index[1])
            ]
        )

        # Get the raw xyz positions (num_nodes, 3) at the prediction index
        y = self.edge_attrs[pred_idx]

        # Convert to torch.Tensor
        x_position = torch.from_numpy(x_position).to(torch.float32)
        edge_index = torch.from_numpy(edge_index).to(torch.long)
        edge_attr = torch.from_numpy(edge_attr).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        #print("x_aminoacid:", self.x_aminoacid.shape)
        #print("x_position:", x_position.shape)
        #print("edge_index:", edge_index.shape)
        #print("edge_attr:", edge_attr.shape)
        #print("y:", y.shape)

        # Construct torch_geometric data object
        data = PairData(
            x_aminoacid=self.x_aminoacid,
            x_position=x_position,
            y=y,
            edge_attr=edge_attr,
            edge_index=edge_index,
        )

        return data



class ContactMapNewDataset(Dataset):
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
        augment_by_reversing_prob: float = 0,
        augment_by_rotating180_prob: float = 0,
        augment_by_translating_prob: float = 0,
        augment_with_noise_mu: float = 0,
        frames_range: Tuple[int, int] = (0, 250000),
        frame_step: int = 1
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
        augment_by_reversing_prob :float, default=0
            With a probability augment_by_reversing_prob, the trajectory is reversed in time
        augment_by_rotating180_prob :float, default=0
            With a probability augment_by_rotating180_prob, rotate x/y/z by 180 degrees about the denter of the box
        augment_by_translating_prob :float, default=0
            With a probability augment_by_rotating180_prob, rotate x/y/z by 180 degrees about the denter of the box
        augment_with_noise_mu :float, default=0
            Add Gaussian noise with mean = 0, standard_deviation = augment_with_noise_mu to each coordinate independently
        frames: tuple(int, int), default=(0, 250000)
            Start and end frame numbers to be read from the trajectory
        frame_step: int, default=1
            Number of frames in the raw trajectory between frames chosen for the window, i.e., using window_size frames every frame_step frames, predict the frame that comes after frame_step frames
        Raises
        ------
        ValueError
            If :obj:`window_size` is greater than 1.
        ValueError
            If the sum of :obj:`window_size` and :obj:`horizon` is longer
            than the input data.
        """

        self._constant_num_node_features = constant_num_node_features
        self.window_size = window_size
        self.horizon = horizon
        # TODO: move augmentations to training routine
        self.augment_by_reversing_prob = augment_by_reversing_prob
        self.augment_by_rotating180_prob = augment_by_rotating180_prob
        self.augment_by_translating_prob = augment_by_translating_prob
        self.augment_with_noise_mu = augment_with_noise_mu
        self.frame_step = frame_step
        self.n_aug = 0
        # TODO: How were these edges chosen? Radial cutoff or Gaussian?
        # TODO: a graph attention mechanism will allow the relevancy of edges to be learnt
        with h5py.File(path, "r", libver="latest", swmr=False) as f:
            # COO formated ragged arrays
            print(f'{path} has {len(f[edge_index_dset_name])} frames')
            self.edge_indices = np.array(f[edge_index_dset_name][frames_range[0]:frames_range[1]])
            self.node_attrs = np.array(f[edge_attr_dset_name][frames_range[0]:frames_range[1]])
            print(f'self.edge_indices: {self.edge_indices.shape}, self.node_attrs: {self.node_attrs .shape}')
            # get the rmsd nums
            try:
                self.rmsd_values = np.array(f['rmsd'][frames_range[0]:frames_range[1]])
            except ValueError as e:
                print("Not able to load rmsd values...")
                self.rmsd_values = []
            if node_feature_dset_name is not None:
                self._node_features_dset = f[node_feature_dset_name][...]
            print(f'self._node_features_dset: {self._node_features_dset.shape}')
        if len(self.edge_indices) - self.window_size - self.horizon + 1 < 0:
            raise ValueError(
                "The sum of window_size and horizon is longer than the input data"
            )

        # Put positions in order (N, num_nodes, 3)
        # These are node attributes that are going to be used as edge attributes later on:
        self.node_attrs = np.transpose(self.node_attrs, [0, 2, 1])
        print(f'self.node_attrs: {self.node_attrs.shape}')
        self.x_aminoacid = self._node_features_dset
        print(f'self.x_aminoacid: {self.x_aminoacid.shape}')
        self.x_aminoacid = torch.from_numpy(self.x_aminoacid).to(torch.long)

    @property
    def num_nodes(self) -> int:
        return self.x_aminoacid.size(0)

    def _compute_node_features(self, node_feature: str) -> np.ndarray: #ContactMapNewDataset
        if node_feature == "constant":
            node_features = np.ones((self.num_nodes, self._constant_num_node_features))
        elif node_feature == "identity":
            node_features = np.eye(self.num_nodes)
        elif node_feature == "amino_acid_onehot":
            node_features = aminoacid_int_to_onehot(self._node_features_dset)
        else:
            # TODO: should use atom type; don't restrict to C-alphas
            raise ValueError(f"node_feature: {node_feature} not supported.")
        return node_features

    def __len__(self): #ContactMapNewDataset
        return len(self.edge_indices) - (self.window_size + self.horizon - 1) * self.frame_step

    def __getitem__(self, idx): #ContactMapNewDataset

        if (self.augment_by_reversing_prob > 0) and (random.random() < self.augment_by_reversing_prob):
            # Reverse trajectory: Given window_size frames after idx, predict frame idx
            pred_idx = idx
            frame_for_edge_attr = idx + (self.window_size + self.horizon - 1) * self.frame_step
            x_position = np.ascontiguousarray(
                np.flip(self.node_attrs[idx + self.horizon * self.frame_step: idx + (self.window_size + self.horizon) * self.frame_step: self.frame_step], 
                        [0])) # reverse time order
            self.n_aug += 1
        else:
            # Forward trajectory: Given window_size frames starting from idx, predict the subsequent frame 
            pred_idx = idx + (self.window_size + self.horizon - 1) * self.frame_step
            frame_for_edge_attr = idx
            x_position = self.node_attrs[idx: idx + self.window_size * self.frame_step: self.frame_step]

        # Get adjacency list
        # TODO: augment by picking frame_for_edge_attr to be any frame in the window just for fixing adjacency list
        edge_index = self.edge_indices[frame_for_edge_attr].reshape(2, -1)  # [2, num_edges]

        # Get the raw xyz positions (num_nodes, 3) at the prediction index
        y = self.node_attrs[pred_idx]
        node_attr_local = self.node_attrs[frame_for_edge_attr].copy()
        if (self.augment_by_translating_prob > 0) and (random.random() < self.augment_by_translating_prob):
            self.n_aug += 1
            translation = np.random.normal(size=[3]) # mean=0, std=1
            x_position += translation
            y += translation
            node_attr_local += translation

        if self.augment_with_noise_mu != 0:
            self.n_aug += 1
            x_position += np.random.normal(loc = 0, scale = self.augment_with_noise_mu, size = x_position.shape)
            node_attr_local += np.random.normal(loc = 0, scale = self.augment_with_noise_mu, size = node_attr_local.shape)
            # Apply noise to inputs not the output:
            #y += np.random.normal(loc = 0, scale = self.augment_with_noise_mu, size = y.shape)

        if (self.augment_by_rotating180_prob > 0):
            # TODO: Need SE(3) transformer or other rotation equivariant model 
            # TODO: implement more general rotation, uniformly sampling the unit sphere
            rot_mat = np.array([1, 1, 1])
            if (random.random() < self.augment_by_rotating180_prob):
                # rotate 180 degrees about x-axis
                rot_mat *= [1, -1, -1]
            if (random.random() < self.augment_by_rotating180_prob):
                # rotate 180 degrees about y-axis
                rot_mat *= [-1, 1, -1]
            if (random.random() < self.augment_by_rotating180_prob):
                # rotate 180 degrees about z-axis            
                rot_mat *= [-1, -1, 1]
            if not np.array_equal(rot_mat, np.array([1, 1, 1])):
                self.n_aug += 1
                #print(rot_mat)
                #print('x', x_position.shape, x_position[0])
                x_position *= rot_mat
                #print('x_', x_position.shape, x_position[0])
                #print('y', y.shape, y[0])
                y *= rot_mat
                #print('y_', y.shape, y[0])
                #print('node_attr_local', node_attr_local.shape, node_attr_local[0])
                node_attr_local *= rot_mat
                #print('node_attr_local_', node_attr_local.shape, node_attr_local[0])

        # Get edge attributes with shape (num_edges, num_edge_features)
        # TODO: edge attribute should be bond type, coordinates should be node attributes; alternatively, try using internal coordinates, radial basis functions, and other standard tricks!
        edge_attr = np.array(
            [
                np.concatenate(
                    (node_attr_local[i, :], node_attr_local[j, :])
                ).flatten()
                for i, j in zip(edge_index[0], edge_index[1])
            ]
        )

        # Convert to torch.Tensor
        x_position = torch.from_numpy(x_position).to(torch.float32)
        edge_index = torch.from_numpy(edge_index).to(torch.long)
        edge_attr = torch.from_numpy(edge_attr).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        #print("x_aminoacid:", self.x_aminoacid.shape)
        #print("x_position:", x_position.shape)
        #print("edge_index:", edge_index.shape)
        #print("edge_attr:", edge_attr.shape)
        #print("y:", y.shape)

        # Construct torch_geometric data object
        data = PairData(
            x_aminoacid=self.x_aminoacid,
            x_position=x_position,
            y=y,
            edge_attr=edge_attr,
            edge_index=edge_index,
        )

        return data # ContactMapNewDataset
