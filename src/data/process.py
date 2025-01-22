from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pymatgen.core import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
from scipy.stats import rankdata
from torch_geometric.utils import dense_to_sparse, add_self_loops
import numpy as np
import torch
from torch.functional import F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import json

class MEGNetDataset(Dataset):
    def __init__(self, root, df, r_max, n_neighbors, name_database):
        """
            root: merupakan tempat menyimpan file data torch_geomtric
            df: merupakan dataframe yang digunakan untuk merubah ke data graph
            database: diisi dengan nama MaterialsProject atau SNUMAT
        """
        self.df = df
        self.r_max = r_max
        self.n_neighbors = n_neighbors
        self.name_database = name_database
        super(MEGNetDataset, self).__init__(root)
    
    @property
    def raw_file_names(self):
        if self.name_database == 'MaterialsProject':
            return ['data_mp.json']
        if self.name_database == 'SNUMAT':
            return ['data_SNUMAT.json']
    
    @property
    def processed_file_names(self):
        return [f'data_{i}' for i in range(0, len(self.df))]
    
    def download(self):
        pass

    def process(self):
        for i, material in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            if self.name_database == 'MaterialsProject':
                structure = Structure.from_str(material['structure'], fmt='cif')
                state = self._get_states_MP(material)
                label = self._get_labels(material['band_gap'])
            elif self.name_database == 'SNUMAT':
                structure = Structure.from_str(material['Structure_rlx'], fmt='poscar')
                state = self._get_states_SNUMAT(material)
                label = self._get_labels(material['Band_gap_HSE'])
            node_features = self._get_node_features(structure)
            edge_index, edge_attr = self._get_edge_features(structure)
            data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        state=state,
                        y=label
                        )
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def _get_node_features(self, structure):
        node_features = [site.specie.number for site in structure]
        node_features = np.array(node_features, dtype=np.int32)
        return torch.tensor(node_features, dtype=torch.int32).view(-1,1)

    def _trimming(self, distance_matrix, r_max, n_neighbors):
        mask = distance_matrix > r_max
        matrix_trimmed = np.ma.array(distance_matrix, mask=mask)
        matrix_trimmed = rankdata(matrix_trimmed, method='ordinal', axis=1)
        matrix_trimmed = np.nan_to_num(np.where(mask, np.nan, matrix_trimmed))
        matrix_trimmed[matrix_trimmed>n_neighbors+1] = 0
        matrix_trimmed = np.where(matrix_trimmed==0, matrix_trimmed, distance_matrix)
        return matrix_trimmed
    
    def _get_edge_features(self, structure):
        distance = structure.distance_matrix
        distance_matrix_trimmed = self._trimming(distance_matrix=distance, r_max=self.r_max, n_neighbors=self.n_neighbors)
        distance_matrix_trimmed = torch.tensor(distance_matrix_trimmed, dtype=torch.float32)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_attr = out[1]
        self_loops = True
        if self_loops == True:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=len(structure), fill_value=0
            )
        return edge_index, edge_attr
    
    def _get_states_MP(self, material):
        return torch.tensor([material['formation_energy_per_atom']], dtype=torch.float32).view(-1,1)
    
    def _get_states_SNUMAT(self, material):
        mapping_dir_indir = {'Direct':0, 'Indirect':1}
        mapping_magnetic_ordering = {'NM':0, 'FM':1, 'AFM':2, 'FiM':3}
        dir_indir = material['Direct_or_indirect_HSE']
        dir_indir = torch.tensor([mapping_dir_indir[dir_indir]])
        one_hot_dir_indir = F.one_hot(dir_indir, num_classes=2).to(torch.float32)
        mag_or = material['Magnetic_ordering']
        mag_or = torch.tensor([mapping_magnetic_ordering[mag_or]])
        one_hot_mag_or = F.one_hot(mag_or, num_classes=4).to(torch.float32)
        state = torch.cat([one_hot_dir_indir, one_hot_mag_or], dim=-1).view(1, -1)
        return state       
    
    def _get_labels(self, label):
        return torch.tensor([label], dtype=torch.float32).view(-1,1)
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
class CGCNNDataset(Dataset):
    def __init__(self, root, df, r_max, n_neighbors, atom_features_dir, name_database):
        """
            root: merupakan tempat menyimpan file data torch_geomtric
            df: merupakan dataframe yang digunakan untuk merubah ke data graph
            database: diisi dengan nama MaterialsProject atau SNUMAT
        """
        self.df = df
        self.r_max = r_max
        self.n_neighbors = n_neighbors
        self.name_database = name_database
        self.atom_features_dir = atom_features_dir
        super(CGCNNDataset, self).__init__(root)
    
    @property
    def raw_file_names(self):
        if self.name_database == 'MaterialsProject':
            return ['data_mp.json']
        if self.name_database == 'SNUMAT':
            return ['data_SNUMAT.json']
    
    @property
    def processed_file_names(self):
        return [f'data_{i}' for i in range(0, len(self.df))]
    
    def download(self):
        pass

    def process(self):
        for i, material in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            if self.name_database == 'MaterialsProject':
                structure = Structure.from_str(material['structure'], fmt='cif')
                label = self._get_labels(material['band_gap'])
            elif self.name_database == 'SNUMAT':
                structure = Structure.from_str(material['Structure_rlx'], fmt='poscar')
                label = self._get_labels(material['Band_gap_HSE'])
            node_features = self._get_node_features(structure)
            edge_index, edge_features = self._get_edge_features(structure)
            data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        y=label
                        )
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def _get_node_features(self, structure):
        with open(self.atom_features_dir, "r") as file:
            atom_features = json.load(file)
        all_node_features = []
        for site in structure:
            atom_number = str(site.specie.number)
            if atom_number in atom_features:
                vector_node = atom_features[atom_number]
                all_node_features.append(vector_node)
        all_node_feats = np.array(all_node_features, dtype=np.int32)
        node_features = torch.tensor(all_node_feats, dtype=torch.int32)
        return node_features
    
    def _trimming(self, distance_matrix, r_max, n_neighbors):
        mask = distance_matrix > r_max
        matrix_trimmed = np.ma.array(distance_matrix, mask=mask)
        matrix_trimmed = rankdata(matrix_trimmed, method='ordinal', axis=1)
        matrix_trimmed = np.nan_to_num(np.where(mask, np.nan, matrix_trimmed))
        matrix_trimmed[matrix_trimmed>n_neighbors+1] = 0
        matrix_trimmed = np.where(matrix_trimmed==0, matrix_trimmed, distance_matrix)
        return matrix_trimmed
    
    def _get_edge_features(self, structure):
        distance = structure.distance_matrix
        distance_matrix_trimmed = self._trimming(distance_matrix=distance, r_max=self.r_max, n_neighbors=self.n_neighbors)
        distance_matrix_trimmed = torch.tensor(distance_matrix_trimmed, dtype=torch.float32)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_attr = out[1]
        self_loops = True
        if self_loops == True:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=len(structure), fill_value=0
            )
        return edge_index, edge_attr
    
    def _get_labels(self, label):
        return torch.tensor([label], dtype=torch.float32).view(-1,1)
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
def datasplit(dataset, 
              train_split, 
              val_split, 
              test_split, 
              random_state,
              batch_size):
    train_idx, temp_idx = train_test_split(range(len(dataset)), 
                                           test_size=(1-train_split),
                                           shuffle=True, 
                                           random_state=random_state)
    
    test_ratio = test_split/(test_split+val_split)

    test_idx, val_idx = train_test_split(temp_idx,
                                         test_size=(1-test_ratio),
                                         shuffle=False,
                                         random_state=random_state)

    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(dataset[test_idx], batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader
