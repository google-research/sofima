# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Elastic image tile stitching.

For elastic stitching, every tile is modeled as a spring-mass mesh (similarly
to a complete section during alignment), and the meshes are packed into a
single array of shape [2, N, y, x] where N is the number of tiles and the 1st
dimension represents the XY position of the mesh node at (x, y).
"""

import enum
import functools as ft
from typing import Dict, Mapping, Tuple

from connectomics.common import bounding_box
import jax
import jax.numpy as jnp
import numpy as np
from sofima import flow_field
from sofima import map_utils

TileXY = Tuple[int, int]
TileFlow = Dict[TileXY, np.ndarray]
TileOffset = Dict[TileXY, Tuple[float, float]]
TileFlowData = Tuple[np.ndarray, TileFlow, TileOffset]


class NeighborInfo(enum.IntEnum):
  """Semantic aliases for indices in the neighbor info array.

  The values here define metadata for a connection between two NN tiles.
  """
  # Neighboring tile index.
  nbor_idx = 0
  # Index within the flow array.
  flow_idx = 1
  # Offset between the two tiles in the dimension orthogonal to the overlap dim,
  # as estimated in stitch_rigid.compute_coarse_offsets (in pixels).
  offset_ortho = 2
  # Size of the flow array along the dimension orthogonal to the overlap dim.
  flow_ortho = 3
  # Size of the flow array along the overlap dimension.
  flow_overlap = 4
  # Components of the offset vector (as returned by compute_flow_map) used
  # to define the relative position of the two tiles when computing the flow
  # field.
  off_x = 5
  off_y = 6
  # Dimension along which the neighboring tile was found (0:x, 1:y).
  dim = 7


def compute_flow_map(tile_map: Mapping[TileXY, np.ndarray],
                     offset_map: np.ndarray,
                     axis: int,
                     patch_size=(120, 120),
                     stride=(20, 20),
                     batch_size=256) -> Tuple[TileFlow, TileOffset]:
  """Computes fine flow between two horizontally adjacent tiles.

  Args:
    tile_map: maps (x, y) tile coordinates to the tile image content
    offset_map: [2, y, x]-shaped array where the vector spanning the first
      dimension is a coarse XY offset between the tiles (x,y) and (x+1,y) or
      (x,y+1)
    axis: axis along which to look for the neighboring tile (0:x, 1:y)
    patch_size: YX patch size in pixels
    stride: YX stride for the flow map in pixels
    batch_size: number of flow vectors to estimate simultaneously

  Returns:
    tuple of dictionaries:
      (x, y) -> flow array
      (x, y) -> xy offset with which the flow was computed
  """
  yx_shape = offset_map.shape[-2:]
  mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
  ret, offsets = {}, {}

  pad_y = patch_size[0] // 2 // stride[0]
  pad_x = patch_size[1] // 2 // stride[1]

  for y in range(0, yx_shape[0] - axis):
    for x in range(0, yx_shape[1] - (1 - axis)):
      if np.isnan(offset_map[0, y, x]):
        continue

      pre = tile_map[x, y]
      post = tile_map[x + (1 - axis), y + axis]

      offset = offset_map[:, y, x]  # off_x, off_y

      # The start coordinate should be aligned to a multiple of stride size.
      rounded_offset = stride[::-1] * np.round(offset / stride[::-1])

      overlap = -int(offset[axis])
      overlap = pre.shape[1 - axis] - ((pre.shape[1 - axis] - overlap) //
                                       stride[1 - axis] * stride[1 - axis])

      # Offset in the direction orthogonal to the overlap.
      ortho_offset = int(rounded_offset[1 - axis])

      pre_sel = list(np.index_exp[:, :])
      post_sel = list(np.index_exp[:, :])
      pre_sel[1 - axis] = np.s_[-overlap:]
      post_sel[1 - axis] = np.s_[:overlap]

      if ortho_offset > 0:  # post is shifted down relative to pre
        pre_sel[axis] = np.s_[ortho_offset:]
        post_sel[axis] = np.s_[:-ortho_offset]
      elif ortho_offset < 0:  # post is shifted up relative to pre
        pre_sel[axis] = np.s_[:ortho_offset]
        post_sel[axis] = np.s_[-ortho_offset:]

      pre = pre[tuple(pre_sel)]
      post = post[tuple(post_sel)]

      f = mfc.flow_field(
          pre, post, patch_size=patch_size, step=stride, batch_size=batch_size)
      # The inverse flow (post, pre) is just -f, so it does not need to be
      # computed separately.

      ret[(x, y)] = np.pad(
          f, [[0, 0], [pad_y, pad_y - 1], [pad_x, pad_x - 1]],
          constant_values=np.nan)
      if axis == 0:
        offsets[x, y] = (-overlap, ortho_offset)
      else:
        offsets[x, y] = (ortho_offset, -overlap)

  return ret, offsets


def aggregate_arrays(
    x_data: TileFlowData, y_data: TileFlowData, tile_map: Mapping[TileXY,
                                                                  np.ndarray],
    coarse_mesh: np.ndarray, stride: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[TileXY, int]]:
  """Aggregates data for all tiles into single arrays.

  Args:
    x_data: tuple of: array of offsets computed between (x, y) and (x + 1, y)
      tiles; dictionary mapping tile coordinates (x, y) to a flow field between
      tiles (x, y) and (x + 1, y); dictionary mapping tile coordinates to an XY
      offset vector used to determine the relative position of the tiles when
      the flow was computed
    y_data: same as x_data, but for (x, y) and (x, y + 1) tiles
    tile_map: maps (x, y) tile coordinates to the tile image content
    coarse_mesh: solution of rigid stitching (see
      stitch_rigid.optimize_coarse_mesh)
    stride: YX stride for the mesh coordinate and flow maps in pixels

  Returns:
    tuple of:
      [2, N, y, x] array of flows between horizontal NN tiles
      [2, M, y, x] array of flows between vertical NN tiles
      [2, n, y, x] array of meshes for all tiles
      [n, 4, 8] array of tile neigbbor metadata; the index in the last dimension
      of this
        array can be interpreted as described in NeighborInfo
      dictionary mapping tile coordinates to a scalar index within the 1st
        dimension of the returned arrays
  """
  cx, fine_x, offsets_x = x_data
  cy, fine_y, offsets_y = y_data

  # Map tile coordinate to a scalar index.
  key_to_idx = {(tx, ty): i for i, (tx, ty) in enumerate(tile_map.keys())}

  # Single arrays for all mesh and flow data. Dims are:
  #  xy, tile index, y_size, x_size
  fx_shape = np.max([v.shape for v in fine_x.values()] + [(2, 1, 1)], axis=0)
  fy_shape = np.max([v.shape for v in fine_y.values()] + [(2, 1, 1)], axis=0)
  fx_all = np.full([2, len(key_to_idx)] + fx_shape[1:].tolist(), np.nan)
  fy_all = np.full([2, len(key_to_idx)] + fy_shape[1:].tolist(), np.nan)

  # Populate flow tables. The individual flow fields can be smaller than the
  # XY size of the array entry. The flow fields always start at (0,0) and
  # are padded with nan's as necessary.
  for k, i in key_to_idx.items():
    # TODO(mjamusz): Consider using the clean_flow function here instead of
    # just dropping the flow metadata channels and using the offset vector
    # (first 2 dimensions) only.
    if k in fine_x:
      f = fine_x[k]
      fx_all[:, i, 0:f.shape[1], 0:f.shape[2]] = f[:2]

    if k in fine_y:
      f = fine_y[k]
      fy_all[:, i, 0:f.shape[1], 0:f.shape[2]] = f[:2]

  # Build a neighbor info table. Dimensions are:
  #   tile_idx, edge_idx, data
  # where the meaning of the position along the last dim is described
  # by NeighborInfo.
  nbors = np.full((len(key_to_idx), 4, 8), -1, dtype=int)
  for tx, ty in tile_map.keys():
    i = key_to_idx[tx, ty]
    if (tx - 1, ty) in fine_x:
      _, yo = cx[:, ty, tx - 1]
      key = flow_key = tx - 1, ty
      ortho, overlap = fine_x[flow_key].shape[-2:]
      off = offsets_x[flow_key]
      nbors[i, 0, :] = (key_to_idx[key], key_to_idx[flow_key], yo, ortho,
                        overlap, off[0], off[1], 0)

    if (tx, ty) in fine_x:
      _, yo = cx[:, ty, tx]
      key = tx + 1, ty
      flow_key = tx, ty
      off = offsets_x[flow_key]
      ortho, overlap = fine_x[flow_key].shape[-2:]
      nbors[i, 1, :] = (key_to_idx[key], key_to_idx[flow_key], yo, ortho,
                        overlap, off[0], off[1], 0)

    if (tx, ty - 1) in fine_y:
      xo, _ = cy[:, ty - 1, tx]
      key = flow_key = tx, ty - 1
      off = offsets_y[flow_key]
      overlap, ortho = fine_y[flow_key].shape[-2:]
      nbors[i, 2, :] = (key_to_idx[key], key_to_idx[flow_key], xo, ortho,
                        overlap, off[0], off[1], 1)

    if (tx, ty) in fine_y:
      xo, _ = cy[:, ty, tx]
      key = tx, ty + 1
      flow_key = tx, ty
      off = offsets_y[flow_key]
      overlap, ortho = fine_y[flow_key].shape[-2:]
      nbors[i, 3, :] = (key_to_idx[key], key_to_idx[flow_key], xo, ortho,
                        overlap, off[0], off[1], 1)

  img_yx = next(iter(tile_map.values())).shape
  box_tile = bounding_box.BoundingBox(
      start=(0, 0, 0), size=(img_yx[1] // stride[1], img_yx[0] // stride[0], 1))
  x_all = np.zeros((2, len(key_to_idx), box_tile.size[1], box_tile.size[0]))

  # The coarse solution forms the initial conditions for fine mesh optimization.
  for tx, ty in tile_map.keys():
    x_all[:, key_to_idx[tx, ty], ...] = coarse_mesh[:, ty, tx].reshape(2, 1, 1)

  return fx_all, fy_all, x_all, nbors, key_to_idx


@ft.partial(jax.jit, static_argnames=['stride', 'dim'])
def _apply_flow(base_mesh: jnp.ndarray, nbor_mesh: jnp.ndarray,
                nbor_flow: jnp.ndarray, mult: int, stride: Tuple[float, float],
                nbor_data: jnp.ndarray, dim: int) -> jnp.ndarray:
  """Updates mesh with data for a neighboring tile.

  Args:
    base_mesh: [2, y, x] mesh of the current tile
    nbor_mesh: [2, y, x] mesh of the other tile
    nbor_flow: [2, n y, x] flow array for all tile pairs
    mult: multiplier for the flow
    stride: yx stride for the flow and mesh data
    nbor_data: [8] array of neighbor info
    dim: 0 if processing a horizontal (x) NN, 1 if vertical (y)

  Returns:
    base_mesh, with part corresponding to neighboring tile updated
  """
  flow_overlap = nbor_data[NeighborInfo.flow_overlap]
  flow_ortho = nbor_data[NeighborInfo.flow_ortho]
  offset_ortho = nbor_data[NeighborInfo.offset_ortho]

  start_par = jnp.where(mult == 1, nbor_mesh.shape[-dim - 1] - flow_overlap, 0)
  start_ortho = jnp.where(
      ((mult == 1) & (offset_ortho > 0)) | ((mult == -1) & (offset_ortho < 0)),
      nbor_mesh.shape[dim - 2] - flow_ortho, 0)

  # yx
  start = jnp.array(
      [
          start_ortho * (1 - dim) + dim * start_par,  #
          start_ortho * dim + (1 - dim) * start_par
      ],
      dtype=int)

  # Compute the updated mesh part by composing the neighbor mesh with
  # the corresponding flow data.
  nbor_flow = mult * jax.lax.dynamic_index_in_dim(
      nbor_flow, nbor_data[NeighborInfo.flow_idx], axis=1, keepdims=False)

  update = map_utils.compose_maps_fast(
      nbor_flow[:, np.newaxis, ...],
      start,
      stride,
      nbor_mesh[:, np.newaxis, ...],
      jnp.zeros_like(start),
      stride,
      mode='constant')[:, 0, ...]
  update += mult * jnp.array([
      nbor_data[NeighborInfo.off_x], nbor_data[NeighborInfo.off_y]
  ]).reshape(2, 1, 1)

  # Paste the updated mesh part into the current mesh.
  tg_start_par = jnp.where(mult == 1, 0,
                           nbor_mesh.shape[-dim - 1] - flow_overlap)
  tg_start_ortho = jnp.where(
      ((mult == 1) & (offset_ortho < 0)) | ((mult == -1) & (offset_ortho > 0)),
      nbor_mesh.shape[dim - 2] - flow_ortho, 0)
  tg_start = (
      0,  #
      tg_start_par * dim + (1 - dim) * tg_start_ortho,
      tg_start_par * (1 - dim) + dim * tg_start_ortho)
  previous = jax.lax.dynamic_slice(base_mesh, tg_start, nbor_flow.shape)
  return jax.lax.dynamic_update_slice(
      base_mesh, jnp.where(jnp.isnan(update), previous, update), tg_start)


@ft.partial(jax.jit, static_argnames=['stride'])
def _update_mesh(mesh: jnp.ndarray,
                 nbor_data: jnp.ndarray,
                 x: jnp.ndarray,
                 fx: jnp.ndarray,
                 fy: jnp.ndarray,
                 stride=(20, 20)) -> jnp.ndarray:
  """Updates mesh with data for a neighboring tile.

  Args:
    mesh: [2, y, x] mesh to update
    nbor_data: [max(NeighborInfo)] array of neighbor info
    x: [2, n, y, x] array of mesh node positions for all tiles
    fx: [2, n, y, x] array of flow data for horizontal tile NNs
    fy: [2, n, y, x] array of flow data for vertical tile NNs
    stride: yx stride for the flow and mesh data

  Returns:
    [2, y, x] updated mesh
  """
  nbor_idx = nbor_data[NeighborInfo.nbor_idx]
  flow_idx = nbor_data[NeighborInfo.flow_idx]

  # -1 -> flow tells us how to move neighbor to match us (prev=us, post=nbor)
  # +1 -> flow tells us how to move us to match neighbor (prev=nbor, post=us)
  mult = jnp.where(nbor_idx == flow_idx, 1, -1)
  nbor_mesh = jax.lax.dynamic_index_in_dim(x, nbor_idx, axis=1, keepdims=False)
  unused = 1
  # pylint: disable=g-long-lambda
  return (jax.lax.cond(
      nbor_idx == -1,  # invalid index?
      lambda _: mesh,  # nothing to update
      lambda _: jax.lax.cond(
          nbor_data[NeighborInfo.dim] == 0,  # horizontal neighbor?
          lambda _: _apply_flow(mesh, nbor_mesh, fx, mult, stride, nbor_data, 0
                               ),
          lambda _: _apply_flow(mesh, nbor_mesh, fy, mult, stride, nbor_data, 1
                               ),
          None),
      None)), unused
  # pylint: enable=g-long-lambda


def compute_target_mesh(nbor_data: jnp.ndarray, x, fx, fy,
                        stride=(20, 20)) -> jnp.ndarray:
  """Computes the target mesh for a tile mesh.

  Given a tile mesh, flow fields can be used to define virtual springs which
  connect some nodes in this mesh to points in the meshes of neighboring tiles.
  This function computes the locations of these points and assembles them into
  an array ("target mesh"). This target mesh can be used to compute forces
  acting on the current tile mesh and pulling it to match the neighboring
  tiles.

  A typical application is to vmap this function over the nbor array, e.g.:
    vmap(partial(compute_target_mesh, x=x, fx=fx, fy=fy))(nbors)

  Args:
    nbor_data: [4, 8] array of neighbor info; -1 in nbor and flow indices
      indicates invalid (missing) entries
    x: [2, n, y, x] array with node positions
    fx: [2, n, y, x] array with flow data for horizontal neighbors
    fy: [2, n, y, x] array with flow data for vertical neighbors
    stride: yx stride for the flow and mesh data

  Returns:
    [2, y, x] array of target positions
  """
  # When used within vmap/jit, dynamic_update_slice with the pasted content
  # extending beyond the updated array will cause the whole update to fail.
  # To mitigate this, extend the buffer sufficiently to ensure that the
  # pasted content (fx, fy) will always fit.
  y_size, x_size = x.shape[-2:]
  y_size += max(fy.shape[-2], fx.shape[-2])
  x_size += max(fy.shape[-1], fx.shape[-1])

  # Scan over neighbors (currently this is always exactly 4 and so
  # could just be explicitly unrolled).
  mesh = jnp.full([2, y_size, x_size], np.nan)
  updated = jax.lax.scan(
      ft.partial(_update_mesh, x=x, fx=fx, fy=fy, stride=stride), mesh,
      nbor_data)[0]

  # Cut the array back to the desired shape.
  return updated[:, :x.shape[-2], :x.shape[-1]]
