import numpy as np
import xarray as xr

from core import util


class _Ocean:
    @property
    def grid_coordinates(self):
        return self.lon, self.lat
    
    @property
    def coastline(self, padding=0, return_indices=False):
        coastline = util.grid.mask_to_contour(self.mask, iterations=padding+1)
        ys, xs = np.where(coastline)
    
        if return_indices:
            return xs, ys
        else:
            return self.lon[xs].data, self.lat[ys].data


class Glazure64(_Ocean):
    def __init__(self, mesh_mask_path):
        mesh_mask = xr.open_dataset(mesh_mask_path)
        
        self.mask = mesh_mask.tmaskutil.data[0].astype(bool)
        self.lon = mesh_mask.nav_lon.data[0]
        self.lat = mesh_mask.nav_lat.data[:,0]
        
    @property
    def boundary_coordinates(self):
        lon_interp = util.grid.get_interpolator(self.lon)
        lat_interp = util.grid.get_interpolator(self.lat)
    
        lon_edge = lon_interp(len(self.lon)-1.5)
        lat_edge = lat_interp(0.5)
    
        return lon_edge, lat_edge
    