import numpy as np
from pathlib import Path
import argparse

from core.data import mesh


class PathIndex:
    def __init__(self, root_dir, dataset_name):
        root_dir = Path(root_dir)
        self.data_dir = root_dir / 'data'
        self.model_dir = root_dir / 'trained'
        self.groundtruth_dir = self.data_dir / 'groundtruth' / dataset_name
        
        self.subsets_dir = self.groundtruth_dir / 'subsets'
        self.density_maps_dir = self.groundtruth_dir / 'density_maps'
        self.subsets_dir = self.groundtruth_dir / 'subsets'
        self.fields_dir = self.groundtruth_dir / 'fields'
        
        self.index_offsets = self.groundtruth_dir / 'index_offsets.npy'
        self.glazure64_mesh_mask = self.data_dir / 'glazure64' / 'mesh_mask.nc'

        
class Loader:
    def __init__(self, path_index):
        self._paths = path_index
        self.glazure64_mesh = mesh.Glazure64(self._paths.glazure64_mesh_mask)
    
    def snapshot_dataset(self, field_name, subset=None, **kwargs):
        from torch.utils.data import Subset
        from core.data import datasets
        
        field_dir = self._paths.fields_dir / field_name
        dataset = datasets.Snapshot(
            field_dir, self._paths.density_maps_dir, self._paths.index_offsets,
            **kwargs)
        
        if subset is not None:
            subset_path = (
                self._paths.subsets_dir / subset).with_suffix('.npy')
            indices = np.load(subset_path, allow_pickle=True)
            dataset = Subset(dataset, indices)
    
        return dataset

 
class ArgParseFormatter(argparse.MetavarTypeHelpFormatter,
                        argparse.ArgumentDefaultsHelpFormatter):
    pass
