from dataclasses import dataclass
from enum import Enum
import numpy as np
import tensorstore as ts

from sofima import stitch_elastic

class CloudStorage(Enum):
    """
    Documented Cloud Storage Options
    """
    S3 = 1
    GCS = 2


@dataclass
class ZarrDataset:
    """
    Parameters for locating Zarr dataset living on the cloud.
    Args:
    cloud_storage: CloudStorage option 
    bucket: Name of bucket
    dataset_path: Path to directory containing zarr files within bucket
    tile_names: List of zarr tiles to include in dataset. 
                Order of tile_names defines an index that 
                is expected to be used in tile_layout.
    tile_layout: 2D array of indices that defines relative position of tiles.
    downsample_exp: Level in image pyramid with each level
                    downsampling the original resolution by 2**downsmaple_exp.
    """

    cloud_storage: CloudStorage
    bucket: str
    dataset_path: str
    tile_names: list[str]
    tile_layout: np.ndarray
    downsample_exp: int


def open_zarr_gcs(bucket: str, path: str) -> ts.TensorStore:
    return ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'gcs',
            'bucket': bucket,
        },
        'path': path,
    }).result()


def open_zarr_s3(bucket: str, path: str) -> ts.TensorStore: 
    return ts.open({
        'driver': 'zarr',
        'kvstore': {
            'driver': 'http',
            'base_url': f'https://{bucket}.s3.us-west-2.amazonaws.com/{path}',
        },
    }).result()


def load_zarr_data(params: ZarrDataset
                   ) -> tuple[list[ts.TensorStore], stitch_elastic.ShapeXYZ]:
    """
    Reads Zarr dataset from input location 
    and returns list of equally-sized tensorstores
    in matching order as ZarrDataset.tile_names and tile size. 
    Tensorstores are cropped to tiles at origin to the smallest tile in the set.
    """
    
    def load_zarr(bucket: str, tile_location: str) -> ts.TensorStore:
        if params.cloud_storage == CloudStorage.S3:
            return open_zarr_s3(bucket, tile_location)
        else:  # cloud == 'gcs'
            return open_zarr_gcs(bucket, tile_location)
    tile_volumes = []
    min_x, min_y, min_z = np.inf, np.inf, np.inf
    for t_name in params.tile_names:
        tile_location = f"{params.dataset_path}/{t_name}/{params.downsample_exp}"
        tile = load_zarr(params.bucket, tile_location)
        tile_volumes.append(tile)
        
        _, _, tz, ty, tx = tile.shape
        min_x, min_y, min_z = int(np.minimum(min_x, tx)), \
                              int(np.minimum(min_y, ty)), \
                              int(np.minimum(min_z, tz))
    tile_size_xyz = min_x, min_y, min_z

    # Standardize size of tile volumes
    for i, tile_vol in enumerate(tile_volumes):
        tile_volumes[i] = tile_vol[:, :, :min_z, :min_y, :min_x]
        
    return tile_volumes, tile_size_xyz


def write_zarr(bucket: str, shape: list, path: str): 
    """ 
    Args: 
    bucket: Name of gcs cloud storage bucket 
    shape: 5D vector in tczyx order, ex: [1, 1, 3551, 576, 576]
    path: Output path inside bucket
    """
    
    return ts.open({
        'driver': 'zarr', 
        'dtype': 'uint16',
        'kvstore' : {
            'driver': 'gcs', 
            'bucket': bucket,
        }, 
        'create': True,
        'delete_existing': True, 
        'path': path, 
        'metadata': {
        'chunks': [1, 1, 128, 256, 256],
        'compressor': {
          'blocksize': 0,
          'clevel': 1,
          'cname': 'zstd',
          'id': 'blosc',
          'shuffle': 1,
        },
        'dimension_separator': '/',
        'dtype': '<u2',
        'fill_value': 0,
        'filters': None,
        'order': 'C',
        'shape': shape,  
        'zarr_format': 2
        }
    }).result()