import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset

from core import util


class Snapshot(Dataset):
    def __init__(self, field_dir, density_map_dir, index_offsets_path,
                 input_map=False, field_interp=0, next_field=False):
        assert 0 <= field_interp <= 1
        
        self._input_map = input_map
        self._field_interp = field_interp
        self._next_field = next_field
        
        # data directories
        self._field_dir = Path(field_dir)
        self._density_map_dir = Path(density_map_dir)

        # index offsets for unravelling from a 1D to 3D index
        index_offsets = np.load(index_offsets_path, allow_pickle=True).item()
        self._time_offsets = index_offsets['time']
        self._ensemble_offsets = index_offsets['ensemble']
        
    def __len__(self):
        return self._time_offsets[-1]

    def __getitem__(self, index):
        # fetch 3D index
        time_index, ensemble_index, obs_index = self.unravel_index(index)
        # fetch input field, input map, and label map
        input_field = self.load_input_field(time_index, obs_index)
        input_map, label_map = self.load_density_map_pair(
            time_index, ensemble_index, obs_index)
        # stack the input field and map
        if input_field.ndim == 2:
            input_field = input_field[None]
        input_data = np.concatenate((input_field, input_map[None]))
        
        if self._input_map:
            return input_data, label_map, input_data[-1]
        return input_data, label_map

    def _load_field_from_index(self, time_index, obs_index):
        field_time_index = str(time_index + obs_index)
        input_field_path = (
            self._field_dir / field_time_index).with_suffix('.npy')
        return np.load(input_field_path)
    
    def load_input_field(self, time_index, obs_index):
        input_field = self._load_field_from_index(time_index, obs_index)
        
        if self._field_interp > 0 or self._next_field:
            input_field_2 = self._load_field_from_index(
                time_index, obs_index+1)
        if self._field_interp > 0:
            input_field = util.misc.interpolated(
                input_field, input_field_2, self._field_interp)
        elif self._next_field:
            input_field = np.concatenate((input_field, input_field_2))
            
        return input_field
    
    def load_density_map_pair(self, time_index, ensemble_index, obs_index):
        start_time_index = str(time_index)
        input_density_path = (
            self._density_map_dir / start_time_index).with_suffix('.nc')
        
        density_maps = xr.open_dataset(input_density_path).density_map
        return density_maps.isel(
            ensemble_id=ensemble_index, obs=[obs_index, obs_index+1]).data
    
    def unravel_index(self, index):
        if index >= len(self):
            raise IndexError('Index out of bounds')
            
        time_index = (index < self._time_offsets).argmax()
        ensemble_index = (
            index < self._ensemble_offsets[time_index]).argmax() - 1
        obs_index = index - self._ensemble_offsets[time_index][ensemble_index]

        return time_index, ensemble_index, obs_index
    
    def random_split(self, proportions):
        assert np.isclose(sum(proportions), 1)
        
        # map 1D indices to 3D indices
        indices_3d = pd.DataFrame(
            [self.unravel_index(i) for i in range(len(self))],
            columns=['time', 'ensemble', 'obs'])
        
        # group indices by ensemble
        ensembles = indices_3d.groupby(['time', 'ensemble'])
        ensemble_indices = ensembles.ngroup()
        n_ensembles = ensembles.ngroups
        
        # randomly assign ensembles to different sets
        ensemble_sets = util.random.split(n_ensembles, proportions)
        
        # lookup the sample indices belonging to each set
        set_indices = []
        for ensemble_set in ensemble_sets:
            within_set = np.isin(ensemble_indices, ensemble_set)
            set_indices.append(np.where(within_set)[0])
        
        return set_indices