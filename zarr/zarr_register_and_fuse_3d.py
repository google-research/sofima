"""Object Wrapper around SOFIMA on Zarr Datasets."""

import functools as ft
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
import time

from connectomics.common import bounding_box
from connectomics.common import box_generator
from connectomics.volume import subvolume
from sofima import coarse_registration, flow_utils, stitch_elastic, stitch_rigid, map_utils, mesh
from sofima.processor import warp

import zarr_io

# NOTE:
# - SOFIMA/ZarrStitcher follows following basis convention:  
# o -- x
# |
# y
# Any reference to 'x' or 'y' adopt this basis. 

# - All displacements are defined in pixel space established 
# by the downsample_exp/resolution of the input images. 


class SyncAdapter:
  """Makes it possible to use a TensorStore objects as a numpy array."""
  
  def __init__(self, tstore):
    self.tstore = tstore

  def __getitem__(self, ind):
    return np.array(self.tstore[ind])

  def __getattr__(self, attr):
    return getattr(self.tstore, attr)

  @property
  def shape(self):
    return self.tstore.shape

  @property
  def ndim(self):
    return self.tstore.ndim


class ZarrFusion(warp.StitchAndRender3dTiles):
    """
    Fusion renderer loading tile data from Zarr.
    """
    cache = {}
    
    def __init__(self, 
                 zarr_params: zarr_io.ZarrDataset,
                 tile_mesh_path: str, 
                 stride_zyx: tuple[int, int, int],
                 offset_xyz: tuple[float, float, float],
                 parallelism=16) -> None:
        super().__init__(zarr_params.tile_layout,
                         tile_mesh_path,
                         "",
                         stride_zyx,
                         offset_xyz, 
                         parallelism)
        self.zarr_params = zarr_params
    
    def _open_tile_volume(self, tile_id: int):
        if tile_id in self.cache:
            return self.cache[tile_id]

        tile_volumes, tile_size_xyz = zarr_io.load_zarr_data(self.zarr_params)
        tile = tile_volumes[tile_id]
        self.cache[tile_id] = SyncAdapter(tile[0,0,:,:,:])
        return self.cache[tile_id]


class ZarrStitcher: 
    """
    Object wrapper around SOFIMA for operating on Zarr datasets.
    """

    def __init__(self,
                 input_zarr: zarr_io.ZarrDataset) -> None:
        """
        zarr_params: See ZarrDataset, params for input dataset 
        """

        self.input_zarr = input_zarr

        self.tile_volumes: list[ts.TensorStore] = []  # 5D tczyx homogenous shape
        self.tile_volumes, self.tile_size_xyz = zarr_io.load_zarr_data(input_zarr)
        self.tile_layout = input_zarr.tile_layout

        self.tile_map: dict[tuple[int, int], ts.TensorStore] = {}
        for y, row in enumerate(tile_layout):
            for x, tile_id in enumerate(row):
                self.tile_map[(x, y)] = self.tile_volumes[tile_id]


    def run_coarse_registration(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs coarse registration. 
        Returns: 
        cx: tile_layout shape
            Each entry represents displacement of current tile towards right neighbor.
        cy: tile_layout shape
            Each entry represents displacement of current tile towards bottom neighbor. 
        coarse_mesh: (3, 1, tile_layout) shape
            Each entry net displacement of current tile. 
        """

        # Custom data loading for coarse registration
        _tile_volumes: list[ts.TensorStore] = []
        for vol in self.tile_volumes:
            _tile_volumes.append(vol.T[:,:,:,0,0])

        cx, cy = coarse_registration.compute_coarse_offsets(self.tile_layout, _tile_volumes)
        coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, 
                                                        cy, 
                                                        mesh_fn=stitch_rigid.elastic_tile_mesh_3d)
        return cx, cy, coarse_mesh


    def run_fine_registration(self, 
                              cx: np.ndarray, 
                              cy: np.ndarray, 
                              coarse_mesh: np.ndarray, 
                              stride_zyx: tuple[int, int, int],
                              save_mesh_path: str = "solved_meshes.npy"
                              ) -> None: 
        """
        Runs fine registration.
        Inputs:
        cx: Coarse offsets in x direction, output of coarse registration.
        cy: Coarse offsets in y direction, output of coarse registration.
        coarse_mesh: Coarse offsets in combined array, output of coarse registration.
        stride_zyx: Subdivision of each tile to create fine mesh. 

        Outputs (inside of output mesh path):
        solved_fine_mesh: Fine mesh containing offsets of each subdivision.
            Shape is (3, tile_index, stride_z, stride_y, stride_x).
        fine_mesh_xy_to_index: Map of tile coordinates to custom mesh tile index.
        stride_zyx: Same as input, by returned as important parameter.
        """
        
        # Custom data loading for fine registration
        _tile_map = {}
        for key, tstore in self.tile_map.items(): 
            _tile_map[key] = SyncAdapter(tstore[0,:,:,:,:])
        
        # INPUT FORMATTING:
        # For axis 0, subtract tile_size x from the offset[0]
        # For axis 1, subtract tile_size y from the offset[1]
        # Tile size is readded inside of stitch_elastic.compute_flow_map3d.
        cx[:, 0, :, :] = cx[:, 0, :, :] - np.array([self.tile_size_xyz[0], 0, 0])
        cy[:, 0, :, :] = cy[:, 0, :, :] - np.array([0, self.tile_size_xyz[1], 0])

        flow_x, offsets_x = stitch_elastic.compute_flow_map3d(_tile_map,
                                                                self.tile_size_xyz, 
                                                                cx, axis=0,
                                                                stride=stride_zyx,
                                                                patch_size=(80, 80, 80))
        flow_y, offsets_y = stitch_elastic.compute_flow_map3d(_tile_map,
                                                                self.tile_size_xyz, 
                                                                cy, axis=1,
                                                                stride=stride_zyx,
                                                                patch_size=(80, 80, 80))

        # Filter patch flows
        kwargs = {"min_peak_ratio": 1.4, "min_peak_sharpness": 1.4, 
                  "max_deviation": 5, "max_magnitude": 0, "dim": 3}
        fine_x = {k: flow_utils.clean_flow(v, **kwargs) for k, v in flow_x.items()}
        fine_y = {k: flow_utils.clean_flow(v, **kwargs) for k, v in flow_y.items()}

        kwargs = {"min_patch_size": 10, "max_gradient": -1, "max_deviation": -1}
        fine_x = {k: flow_utils.reconcile_flows([v], **kwargs) for k, v in fine_x.items()}
        fine_y = {k: flow_utils.reconcile_flows([v], **kwargs) for k, v in fine_y.items()}

        # Update mesh (convert coarse tile mesh into fine patch mesh)
        data_x = (cx[:, 0, ...], fine_x, offsets_x)
        data_y = (cy[:, 0, ...], fine_y, offsets_y)
        fx, fy, fine_mesh, nbors, fine_mesh_xy_to_index = stitch_elastic.aggregate_arrays(
            data_x, data_y, list(self.tile_map.keys()),
            coarse_mesh[:, 0, ...], stride=stride_zyx, tile_shape=self.tile_size_xyz[::-1])

        @jax.jit
        def prev_fn(x):
            target_fn = ft.partial(stitch_elastic.compute_target_mesh, 
                                   x=x, fx=fx, fy=fy, stride=stride_zyx)
            x = jax.vmap(target_fn)(nbors)
            return jnp.transpose(x, [1, 0, 2, 3, 4])

        config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=0.01, k=0.1, stride=stride_zyx,
                                        num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                        dt_max=100, prefer_orig_order=False,
                                        start_cap=0.1, final_cap=10., remove_drift=True)

        solved_fine_mesh, ekin, t = mesh.relax_mesh(fine_mesh, None, config, 
                                                    prev_fn=prev_fn, mesh_force=mesh.elastic_mesh_3d)

        # Save the mesh/mesh index map 
        np.savez_compressed(save_mesh_path, 
                            x=solved_fine_mesh, 
                            key_to_idx=fine_mesh_xy_to_index, 
                            stride_zyx=stride_zyx)


    def run_fusion(self, 
                    output_cloud_storage: zarr_io.CloudStorage,
                    output_bucket: str, 
                    output_path: str,
                    downsample_exp: int,
                    cx: np.ndarray, 
                    cy: np.ndarray, 
                    tile_mesh_path: str, 
                    parallelism: int = 16
                    ) -> None: 
        """
        Runs fusion. 
        Inputs: 
        output_cloud_storage, output_bucket, output_path: 
            Output storage parameters  
        downsample_exp: 
            Desired output resolution, 0 for highest resolution.
        fine_mesh, fine_mesh_xy_to_index, stride_zyx:
            Fine mesh offsets and accompanying metadata, 
            output of coarse/fine registration.
        parallelism: 
            Multithreading. 
        """

        data = np.load(tile_mesh_path)
        fine_mesh = data['x']
        fine_mesh_xy_to_index = data['key_to_idx']
        stride_zyx = data['stride_zyx']

        if output_cloud_storage == zarr_io.CloudStorage.S3:
            raise NotImplementedError(
                'TensorStore does not support s3 writes.'
            )

        fusion_zarr = self.input_zarr
        fusion_mesh = fine_mesh
        fusion_stride_zyx = stride_zyx
        fusion_tile_size_zyx = self.tile_size_xyz[::-1]
        if downsample_exp != self.input_zarr.downsample_exp:
            # Reload the data at target resolution
            fusion_zarr = zarr_io.ZarrDataset(self.input_zarr.cloud_storage,
                                                self.input_zarr.bucket,
                                                self.input_zarr.dataset_path, 
                                                self.input_zarr.tile_names,
                                                downsample_exp)

            # Rescale fine mesh, stride
            curr_exp = self.input_zarr.downsample_exp
            target_exp = downsample_exp
            scale_factor = 2**(curr_exp - target_exp)
            fusion_mesh = fine_mesh * scale_factor
            fusion_stride_zyx = tuple(np.array(stride_zyx) * scale_factor)
            fusion_tile_size_zyx = tuple(np.array(self.tile_size_xyz)[::-1] * scale_factor)
            print(f'{scale_factor=}')

        start = np.array([np.inf, np.inf, np.inf])
        map_box = bounding_box.BoundingBox(
            start=(0, 0, 0),
            size=fusion_mesh.shape[2:][::-1],
        )
        fine_mesh_index_to_xy = {
            v: k for k, v in fine_mesh_xy_to_index.items()
        }
        for i in range(0, fusion_mesh.shape[1]): 
            tx, ty = fine_mesh_index_to_xy[i]
            mesh = fusion_mesh[:, i, ...]
            tg_box = map_utils.outer_box(mesh, map_box, fusion_stride_zyx)

            out_box = bounding_box.BoundingBox(
                start=(
                tg_box.start[0] * fusion_stride_zyx[2] + tx * fusion_tile_size_zyx[2],
                tg_box.start[1] * fusion_stride_zyx[1] + ty * fusion_tile_size_zyx[1],
                tg_box.start[2] * fusion_stride_zyx[0],
                ),
                size=(
                tg_box.size[0] * fusion_stride_zyx[2],
                tg_box.size[1] * fusion_stride_zyx[1],
                tg_box.size[2] * fusion_stride_zyx[0],
                )
            )
            start = np.minimum(start, out_box.start)
            print(f'{tg_box=}')
            print(f'{out_box=}')

        crop_offset = -start
        print(f'{crop_offset=}')

        # Fused shape
        cx[np.isnan(cx)] = 0    
        cy[np.isnan(cy)] = 0
        x_overlap = cx[2,0,0,0] / self.tile_size_xyz[0]
        y_overlap = cy[1,0,0,0] / self.tile_size_xyz[1]
        y_shape, x_shape = cx.shape[2], cx.shape[3]

        fused_x = fusion_tile_size_zyx[2] * (1 + ((x_shape - 1) * (1 - x_overlap)))
        fused_y = fusion_tile_size_zyx[1] * (1 + ((y_shape - 1) * (1 - y_overlap)))
        fused_z = fusion_tile_size_zyx[0]
        fused_shape_5d = [1, 1, int(fused_z), int(fused_y), int(fused_x)]
        print(f'{fused_shape_5d=}')

        # INPUT FORMATTING: 
        # Save rescaled mesh back into .npz volume
        # as this is the expected input of warp.StitchAndRender3dTiles.process
        rescaled_mesh_path = 'rescaled_fusion_mesh.npz'
        np.savez_compressed(rescaled_mesh_path,
                            x=fusion_mesh,
                            key_to_idx=fine_mesh_xy_to_index, 
                            stride_zyx=fusion_stride_zyx)

        # Perform fusion
        ds_out = zarr_io.write_zarr(output_bucket, fused_shape_5d, output_path)
        renderer = ZarrFusion(zarr_params=fusion_zarr, 
                              tile_mesh_path=rescaled_mesh_path,
                              stride_zyx=fusion_stride_zyx,
                              offset_xyz=crop_offset, 
                              parallelism=parallelism)

        box = bounding_box.BoundingBox(start=(0,0,0), size=ds_out.shape[4:1:-1])  # Needs xyz 
        gen = box_generator.BoxGenerator(box, (512, 512, 512), (0, 0, 0), True) # These are xyz
        renderer.set_effective_subvol_and_overlap((512, 512, 512), (0, 0, 0))
        for i, sub_box in enumerate(gen.boxes):
            t_start = time.time()

            # Feed in an empty subvol, with dimensions of sub_box. 
            inp_subvol = subvolume.Subvolume(np.zeros(sub_box.size[::-1], dtype=np.uint16)[None, ...], sub_box)
            ret_subvol = renderer.process(inp_subvol)  # czyx

            t_render = time.time()

            # ret_subvol is a 4D CZYX volume
            slice = ret_subvol.bbox.to_slice3d()
            slice = (0, 0, slice[0], slice[1], slice[2])
            ds_out[slice].write(ret_subvol.data[0, ...]).result()
            
            t_write = time.time()
            
            print('box {i}: {t1:0.2f} render  {t2:0.2f} write'.format(i=i, t1=t_render - t_start, t2=t_write - t_render))


    # TODO fix this too 
    def run_fusion_on_coarse_mesh(self, 
                                  output_cloud_storage: zarr_io.CloudStorage,
                                  output_bucket: str, 
                                  output_path: str,
                                  downsample_exp: int,
                                  cx: np.ndarray, 
                                  cy: np.ndarray,
                                  coarse_mesh: np.ndarray,
                                  stride_zyx: tuple[int, int, int] = (20, 20, 20),
                                  save_mesh_path: str = "solved_meshes.npy",
                                  parallelism: int = 16) -> None:
        """
        Transforms coarse mesh into fine mesh before 
        passing along to ZarrStitcher._run_fusion(...)
        """
 
        # Create Fine Mesh Tile Index
        fine_mesh_xy_to_index = {(tx, ty): i for i, (tx, ty) in enumerate(self.tile_map.keys())}

        # Convert Coarse Mesh into Fine Mesh
        dim = len(stride_zyx)
        mesh_shape = (np.array(self.tile_size_xyz[::-1]) // stride_zyx).tolist()
        fine_mesh = np.zeros([dim, len(fine_mesh_xy_to_index)] + mesh_shape, dtype=np.float32)
        for (tx, ty) in self.tile_map.keys(): 
            fine_mesh[:, fine_mesh_xy_to_index[tx, ty], ...] = coarse_mesh[:, 0, ty, tx].reshape(
            (dim,) + (1,) * dim)
    
        # Save the mesh/mesh index map 
        np.savez_compressed(save_mesh_path, 
                            x=fine_mesh, 
                            key_to_idx=fine_mesh_xy_to_index, 
                            stride_zyx=stride_zyx)

        self.run_fusion(output_cloud_storage,
                        output_bucket, 
                        output_path,
                        downsample_exp,
                        cx, 
                        cy, 
                        save_mesh_path,
                        parallelism)


if __name__ == '__main__':
    # Example set of Application Inputs 
    cloud_storage = zarr_io.CloudStorage.S3
    bucket = 'aind-open-data'
    dataset_path = 'diSPIM_647459_2022-12-07_00-00-00/diSPIM.zarr'
    downsample_exp = 2
    tile_names = ['tile_X_0000_Y_0000_Z_0000_CH_0405_cam1.zarr', 
                    'tile_X_0001_Y_0000_Z_0000_CH_0405_cam1.zarr']
    tile_layout = np.array([[1],
                            [0]])
    input_zarr = zarr_io.ZarrDataset(cloud_storage=cloud_storage,
                                bucket=bucket,
                                dataset_path=dataset_path, 
                                tile_names=tile_names,
                                tile_layout=tile_layout,
                                downsample_exp=downsample_exp)

    # Application Outputs
    output_cloud_storage = zarr_io.CloudStorage.GCS
    output_bucket = 'YOUR-BUCKET-HERE'
    output_path = 'YOUR-OUTPUT-NAME.zarr' 

    # Processing
    save_mesh_path = 'solved_mesh.npy'
    zarr_stitcher = ZarrStitcher(input_zarr)
    cx, cy, coarse_mesh = zarr_stitcher.run_coarse_registration()
    zarr_stitcher.run_fine_registration(cx, 
                                        cy, 
                                        coarse_mesh, 
                                        stride_zyx=(20, 20, 20),
                                        save_mesh_path=save_mesh_path)
    zarr_stitcher.run_fusion(output_cloud_storage=output_cloud_storage,
                            output_bucket=output_bucket,
                            output_path=output_path,
                            downsample_exp=0,  # For full resolution fusion. 
                            cx=cx,
                            cy=cy,
                            tile_mesh_path=save_mesh_path)