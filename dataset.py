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
#PathLike = Union[str, Path]
import glob
import math

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
        path: str,
        edge_index_dset_name: str = "contact_map",
        edge_attr_dset_name: str = "point_cloud",
        node_feature_dset_name: Optional[str] = "amino_acids",
        node_feature: str = "amino_acid_onehot",
        constant_num_node_features: int = 20,
        window_size: int = 1,
        horizon: int = 1,
        n_frames: int = 250000,
        node_feature_dset_path: str = None
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

        if str(path)[-3:] == '.h5':
            # only process one file
            with h5py.File(path, "r", libver="latest", swmr=False) as f:
                # COO formated ragged arrays
                self.edge_indices = np.array(f[edge_index_dset_name][:n_frames])
                self.node_attrs = np.array(f[edge_attr_dset_name][:n_frames])
                # get the rmsd nums
                try:
                    self.rmsd_values = np.array(f['rmsd'][:n_frames])
                except ValueError as e:
                    print("Not able to load rmsd values...")
                    self.rmsd_values = []
                if node_feature_dset_name is not None:
                    if node_feature_dset_path is not None:
                        with h5py.File(node_feature_dset_path, "r", libver="latest", swmr=False) as node_file:
                            self._node_features_dset = node_file[node_feature_dset_name][...]
                    else:
                        self._node_features_dset = f[node_feature_dset_name][...]

        else:
            self.edge_indices = []
            self.node_attrs = []
            # process each file
            h5_files = glob.glob(str(path)+'/*.h5')
            h5_files.sort()
            for i in h5_files:
                with h5py.File(i, "r", libver="latest", swmr=False) as f:
                    # COO formated ragged arrays
                    self.edge_indices.extend(list(f[edge_index_dset_name][:n_frames]))
                    self.node_attrs.extend(list(f[edge_attr_dset_name][:n_frames]))

            with h5py.File(h5_files[0], "r", libver="latest", swmr=False) as f:
                try:
                    self.rmsd_values = np.array(f['rmsd'][:n_frames])
                except ValueError as e:
                    print("Not able to load rmsd values...")
                    self.rmsd_values = []
                if node_feature_dset_name is not None:
                    if node_feature_dset_path is not None:
                        with h5py.File(node_feature_dset_path, "r", libver="latest", swmr=False) as node_file:
                            self._node_features_dset = np.array(node_file[node_feature_dset_name][...])
                    else:
                        self._node_features_dset = f[node_feature_dset_name][...]



        if len(self.edge_indices) - self.window_size - self.horizon + 1 < 0:
            raise ValueError(
                "The sum of window_size and horizon is longer than the input data"
            )

        # Put positions in order (N, num_nodes, 3)
        self.node_attrs = np.transpose(self.node_attrs, [0, 2, 1])
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
        x_position = self.node_attrs[idx:idx+self.window_size]
        # x_position = self.node_attrs[idx]

        # Get adjacency list
        edge_index = self.edge_indices[idx].reshape(2, -1)  # [2, num_edges]

        # Get edge attributes with shape (num_edges, num_edge_features)
        # Each edge attribute is the positions of both atoms A,B
        # And looks like [Ax, Ay, Az, Bx, By, Bz]
        edge_attr = np.array(
            [
                np.concatenate(
                    (self.node_attrs[idx, i, :], self.node_attrs[idx, j, :])
                ).flatten()
                for i, j in zip(edge_index[0], edge_index[1])
            ]
        )

        # Get the raw xyz positions (num_nodes, 3) at the prediction index
        y = self.node_attrs[pred_idx]

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
        path: str,
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
        frame_step: int = 1,
        node_feature_dset_path: str = None,
        residue_step: int = 1,
        split_frac: float = 1.0,
        traj_id = None
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
        residue_step: int, default=1
            Only use every residue_step residue from each frame of the trajectory
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
        #self.sub_traj_length = frames_range[1] - frames_range[0]
        self.boundary_frames = []
        self.n_traj = 0
        if str(path)[-3:] == '.h5':
            # only process one file
            with h5py.File(path, "r", libver="latest", swmr=False) as f:
                # COO formated ragged arrays
                print(f'{path} has {len(f[edge_index_dset_name])} frames')
                self.edge_indices = np.array(f[edge_index_dset_name][frames_range[0]:frames_range[1]])
                self.node_attrs = np.array(f[edge_attr_dset_name][frames_range[0]:frames_range[1]]) # (4800, 6091, 3)
                # get the rmsd nums
                try:
                    self.rmsd_values = np.array(f['rmsd'][frames_range[0]:frames_range[1]])
                except ValueError as e:
                    print("Not able to load rmsd values...")
                    self.rmsd_values = []
                if node_feature_dset_name is not None:
                    if node_feature_dset_path is not None:
                        with h5py.File(node_feature_dset_path, "r", libver="latest", swmr=False) as node_file:
                            self._node_features_dset = node_file[node_feature_dset_name][...]
                    else:
                        self._node_features_dset = f[node_feature_dset_name][...]
        else:       
            self.edge_indices = []
            self.node_attrs = []
            # process each file
            def ff(x):
                #print('*', x, '*')
                parts = x.split('.')
                if len(parts) == 3: return [int(parts[-2]), 0]
                else: return [int(parts[-3]), int(parts[-2])]
            h5_files = sorted(glob.glob(str(path)+'/*.h5'), key=ff) #lambda x: [int(x.split('.')[-3]), int(x.split('.')[-2])])
            print(f'Found {len(h5_files)} traj segments in {path}')
            if traj_id is not None:
                h5_files = [h5_file for h5_file in h5_files if f'eq.{traj_id}.' in h5_file]
                print(f'Reading all frames from {len(h5_files)} files belonging to traj {traj_id}')
                frames_range = [None, None]
            elif split_frac > 0:
                n_files = max(1, math.floor(split_frac * len(h5_files)))
                h5_files = h5_files[:n_files]
                print(f'Reading all frames from the first {n_files} h5 files')
                frames_range = [None, None]
            elif split_frac < 0:
                n_files = min(-1, math.floor(split_frac * len(h5_files)))
                h5_files = h5_files[n_files:]
                print(f'Reading all frames from the last {n_files} h5 files')
                frames_range = [None, None]

            with h5py.File(h5_files[0], "r", libver="latest", swmr=False) as f:
                #try:
                #    self.rmsd_values = np.array(f['rmsd'][frames_range[0]:frames_range[1]])
                #except ValueError as e:
                #    print("Not able to load rmsd values...")
                #    self.rmsd_values = []
                if node_feature_dset_name is not None:
                    if node_feature_dset_path is not None:
                        with h5py.File(node_feature_dset_path, "r", libver="latest", swmr=False) as node_file:
                            self._node_features_dset = np.array(node_file[node_feature_dset_name][...])
                    else:
                        self._node_features_dset = f[node_feature_dset_name][...]
                        #self._node_features_dset = np.array([0] * len(f[edge_attr_dset_name][0]))

            #h5_files.sort()
            traj_length = {}
            n_total_frames = 0
            self.rmsd_values = []
            for i in h5_files:
                with h5py.File(i, "r", libver="latest", swmr=False) as f:
                    # COO formated ragged arrays
                    n_frames = len(f[edge_attr_dset_name][frames_range[0]:frames_range[1]])
                    print(f'{i} has {len(f[edge_index_dset_name])} frames, using {n_frames}', type(f[edge_index_dset_name]), type(f[edge_attr_dset_name]))
                    self.edge_indices.extend(list(f[edge_index_dset_name][frames_range[0]:frames_range[1]]))
                    self.node_attrs.extend(list(f[edge_attr_dset_name][frames_range[0]:frames_range[1]]))
                    self.rmsd_values.extend(f['rmsd'][frames_range[0]:frames_range[1]])
                    parts = i.split('.')
                    if len(parts) == 3: traj_id = int(parts[-2])
                    else: traj_id = int(parts[-3])
                    if traj_id not in traj_length: 
                        traj_length[traj_id] = n_frames
                        if n_total_frames > 0:
                            self.boundary_frames.append(n_total_frames)
                    else: traj_length[traj_id] += n_frames
                    n_total_frames += n_frames
                    self.n_traj += 1
            self.rmsd_values = np.array(self.rmsd_values)
            print('rmsd values', len(self.rmsd_values))


        for traj_id in traj_length:
            if traj_length[traj_id] < (self.window_size + self.horizon - 1) * self.frame_step:
                raise ValueError(
                    f"Sub-trajectory {traj_id} with {traj_length[traj_id]} frames) is shorter than (window_size ({self.window_size}) + horizon ({self.horizon}) - 1) * frame_step"\
                    f" ({self.frame_step}): cannot construct any examples!"
                )
        print(f'{self.n_traj} trajectories; boundary frames: {self.boundary_frames}')

        # Put positions in order (N, num_nodes, 3)
        # These are node attributes that are going to be used as edge attributes later on:
        self.node_attrs = np.transpose(self.node_attrs, [0, 2, 1])
        print(f'self.node_attrs: {self.node_attrs.shape}') #  (4800, 6091, 3)
        self.x_aminoacid = self._node_features_dset
        print(f'self.x_aminoacid: {len(self.x_aminoacid)}') # (6091,)
        self.x_aminoacid = torch.from_numpy(self.x_aminoacid).to(torch.long) 
        print(f'self.edge_indices: {len(self.edge_indices)} x {self.edge_indices[0].shape}', self.edge_indices[0]) # 4800 x (125850,) 
        self.num_residues = len(self.x_aminoacid)
        self.num_frames = len(self.node_attrs)

        if residue_step > 1:
            self.residue_step = residue_step
            print(f'residue_step={residue_step}')
            self.node_attrs = self.node_attrs[:, range(0, self.num_residues, self.residue_step), :]
            print(f'self.node_attrs: {self.node_attrs.shape}') #   (160, 610, 3)
            self.x_aminoacid = self.x_aminoacid[range(0, self.num_residues, self.residue_step)]
            print(f'self.x_aminoacid: {self.x_aminoacid.shape}') # (6091,)
            self.edge_indices = [
                list(np.array([
                    [i // self.residue_step, j // self.residue_step]
                    for i, j in edge_index.reshape(-1, 2)
                    if (i % self.residue_step == 0) and (j % self.residue_step == 0)
                ]).flatten())
                for edge_index in self.edge_indices
            ]
            self.num_residues = len(self.x_aminoacid)
            print(f'self.edge_indices: {len(self.edge_indices)} x {len(self.edge_indices[0])}')#, self.edge_indices[0]) # 4800 x (125850,) 
            hf = h5py.File(f'all_frames{self.num_frames}_residues{self.num_residues}.h5', 'w')
            print(node_feature_dset_name, type(self.node_attrs))
            hf.create_dataset(node_feature_dset_name, data=self.node_attrs)
            print(edge_index_dset_name, type(self.edge_indices))

            dt = h5py.vlen_dtype(np.dtype('int32'))
            dset = hf.create_dataset(edge_index_dset_name, (len(self.edge_indices),), dtype=dt) #, data=self.edge_indices)
            for i, edge_index in enumerate(self.edge_indices):
                dset[i] = edge_index
            hf.close()
            sys.exit(0)

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

    def get_traj_of_frame(self, frame):
        for traj, bf in enumerate(self.boundary_frames):
            if frame < bf: return traj
        return self.n_traj - 1

    def __getitem__(self, idx): #ContactMapNewDataset
        # The set of frames should not span different trajectories, so move idx if it is too close to the frame where differnet trajectories have been stitched together.
        traj_start = self.get_traj_of_frame(idx) # % self.sub_traj_length
        traj_end = self.get_traj_of_frame(idx + (self.window_size + self.horizon - 1) * self.frame_step) #% self.sub_traj_length
        if traj_start != traj_end:
            # Picking a random frame in traj_start
            # Note: this will affect validation
            #idx = random.randrange(traj_start * self.sub_traj_length, (traj_start + 1) * self.sub_traj_length)
            idx -= (self.window_size + self.horizon - 1) * self.frame_step

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
