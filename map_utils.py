# coding=utf-8
# Copyright 2022-2023 The Google Research Authors.
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
"""Utilities for manipulating coordinate maps stored as volumes.

A volume-backed coordinate map associates the (x, y[, z]) voxel coordinates
in the volume with new (u, v[, w]) coordinates. This is done by storing the
offset relative to (x, y[, z]) as (Δx, Δy[, Δz]) in the two (three) channels
of the volume, so that the mapping is:

  x -> u = x + Δx
  y -> v = y + Δy
 [z -> w = z + Δz]

Even though the physical representation is identical, conceptually we
can distinguish *forward* and *inverse* maps. A forward map associates
a point (x, y) with its new location (u, v) as described above. An
inverse map provides the original location (x, y) for every target
point (u, v).

Coordinate maps can also be stored in absolute format, associating the
absolute coordinates (u, v) directly with every point in the map. We only
use these absolute maps for in-memory manipulation and never write
them out to disk. The relative representation reduces precision loss
and keeps precision loss constant across space as along as the source
and target points do not lie far away from each other.

Invalid values in coordinate maps are indicated by nan's.

The size of the first dimension defines the dimensionality of the space
to which the coordinate map applies (2d or 3d). When a 2d coordinate
map is stored in a 4d array (c, z, y, x), it is interpreted as a collection
of independent coordinate maps, indexed by the 'z' dimension.

The following properties hold for composition (comp) and warping (warp)
with coordinate maps:
  comp(a, b)^-1 = comp(b^-1, a^-1)
  warp(img, comp(a, b)) = warp(warp(img, b), a)
where x^-1 indicates the inverse of x.

TODO(mjanusz): Clean up stride format.
"""

import collections
from typing import Sequence
from connectomics.common import bounding_box
import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy import spatial


StrideZYX = float | Sequence[float]
ShapeZYX = tuple[int, int] | tuple[int, int, int]


def _interpolate_points(
    data_points: Sequence[np.ndarray],
    query_points: Sequence[np.ndarray],
    *values,
    method: str = 'linear',
) -> np.ndarray:
  """Interpolates Nd data.

  This is like griddata(), but for multi-dimensional fields (defined by
  *values).

  Args:
    data_points: arrays of x, y, ... coordinates where the field components are
      defined
    query_points: arrays of x, y, ... coordinates at which to interpolate data
    *values: one or more scalar component fields to interpolate
    method: interpolation scheme to use (linear, nearest, cubic)

  Returns:
    components of the field sampled at 'query_points' as an ndarray; shape
    [len(values)] + shape(query_points)
  """
  if len(data_points) != len(query_points):
    raise ValueError(
        'Data and query points dimensionalities needs to match, are: '
        f'{len(data_points)} and {len(query_points)}'
    )

  if method == 'nearest':
    ip = interpolate.NearestNDInterpolator(data_points, values[0])
    ret = []
    ret.append(ip(query_points))
    for val in values[1:]:
      ip.values = val
      ret.append(ip(query_points))

    return np.array(ret)

  assert method in ('linear', 'cubic')
  data_points = np.array(data_points).T
  tri = spatial.Delaunay(np.ascontiguousarray(data_points, dtype=np.double))

  values = np.array(values).T  # [N, dim]
  if method == 'linear':
    ip = interpolate.LinearNDInterpolator(tri, values, fill_value=np.nan)
  else:
    ip = interpolate.CloughTocher2DInterpolator(tri, values, fill_value=np.nan)
  return ip(query_points).T


def _as_vec(value: StrideZYX, dim: int) -> Sequence[float]:
  if not isinstance(value, collections.abc.Sequence):
    return (value,) * dim

  assert len(value) == dim
  return value


def _identity_map_absolute(
    coord_shape: ShapeZYX,
    stride: StrideZYX,
) -> list[np.ndarray]:
  """Generates an identity map in absolute form.

  Args:
    coord_shape: [z, ]y, x shape of the map to generate
    stride: distance between nearest neighbors of the coordinate map ([z]yx
      sequence or a single float)

  Returns:
    identity maps: [z -> z,] y -> y, x -> x
  """
  dim = len(coord_shape)
  stride = _as_vec(stride, dim)
  return [
      hx * step
      for hx, step in zip(np.mgrid[[np.s_[:s] for s in coord_shape]], stride)
  ]


def to_absolute(
    coord_map: np.ndarray,
    stride: StrideZYX,
    box: bounding_box.BoundingBox | None = None,
) -> np.ndarray:
  """Converts a coordinate map from relative to absolute representation.

  Args:
    coord_map: [2 or 3, z, y, x] array of coordinates, where the channels
      represent a (Δx, Δy[, Δz]) offset from the original (x, y[, z]) location
    stride: distance between nearest neighbors of the coordinate map ([z]yx
      sequence or a single float)
    box: bounding box from which coord_map was extracted; if not provided, the
      returned coordinates will have origin at the beginning of coord_map

  Returns:
    coordinate map where entries represent new (u, v) locations in the
    global coordinate system
  """
  coord_map = coord_map.copy()
  dim = coord_map.shape[0]
  stride = _as_vec(stride, dim)
  off_zyx = _identity_map_absolute(coord_map.shape[-dim:], stride)

  if box is not None:
    if not np.all(coord_map.shape[-dim:][::-1] == box.size[:dim]):
      raise ValueError(
          f'box shape ({box.size}) mismatch with coord map ({coord_map.shape})'
      )
    off_zyx = [
        o + start * step
        for o, step, start in zip(off_zyx, stride, box.start[:dim][::-1])
    ]

  for i in range(dim):
    coord_map[i, ...] += off_zyx[-(i + 1)]

  return coord_map


def to_relative(
    coord_map: np.ndarray,
    stride: StrideZYX,
    box: bounding_box.BoundingBox | None = None,
) -> np.ndarray:
  """Converts a coordinate map from absolute to relative representation.

  Args:
    coord_map: [2 or 3, z, y, x] array of coordinates, where the channels
      represent an absolute (x, y[, z]) location in space
    stride: distance between nearest neighbors of the coordinate map ([z]yx
      sequence or a single float)
    box: bounding box from which coord_map was extracted

  Returns:
    coordinate map where entries represent a (Δx, Δy[, Δz]) offset from the
    original (x, y[, z]) location
  """
  coord_map = coord_map.copy()
  dim = coord_map.shape[0]
  stride = _as_vec(stride, dim)
  off_zyx = _identity_map_absolute(coord_map.shape[-dim:], stride)
  if box is not None:
    if not np.all(coord_map.shape[-dim:][::-1] == box.size[:dim]):
      raise ValueError(
          f'box shape ({box.size}) mismatch with coord map ({coord_map.shape})'
      )

    for i in range(dim):
      off_zyx[-(i + 1)] += box.start[i] * stride[-(i + 1)]

  for i in range(dim):
    coord_map[i, ...] -= off_zyx[-(i + 1)]

  return coord_map


def fill_missing(
    coord_map: np.ndarray,
    *,
    extrapolate=False,
    invalid_to_zero=False,
    interpolate_first=True,
) -> np.ndarray:
  """Fills missing entries in a coordinate map.

  Args:
    coord_map: [2 or 3, z, y, x] coordinate map in relative format
    extrapolate: if False, will only fill by interpolation
    invalid_to_zero: whether to zero out completely invalid sections (i.e.,
      reset to identity map)
    interpolate_first: whether to try interpolation before extrapolation

  Returns:
    coordinate map with invalid entries replaced by interpolated/
    extrapolated values
  """
  if not np.any(np.isnan(coord_map)):
    return coord_map

  s = coord_map.shape
  dim = s[0]
  if dim == 2:
    query_coords = np.mgrid[: s[-2], : s[-1]]  # yx
  elif dim == 3:
    query_coords = np.mgrid[: s[-3], : s[-2], : s[-1]]  # zyx

  query_points = tuple([q.ravel() for q in query_coords[::-1]])  # xy[z]

  rets = []

  def _process_map(curr_map):
    ret = curr_map.copy()
    valid = np.all(np.isfinite(curr_map), axis=0)
    if not np.any(valid):
      if invalid_to_zero:
        ret[...] = 0
        return ret

    points = tuple([q[valid] for q in query_coords[::-1]])  # xy[z]

    if interpolate_first:
      try:
        intp = _interpolate_points(
            points, query_points, *[c[valid] for c in curr_map]
        )
        for i, update in enumerate(intp):
          ret[i, ...] = update.reshape(s[1:])
      except spatial.qhull.QhullError:
        pass

    # It would be nice to do extrapolation with RBFs here, but as of
    # early 2020, the scipy implementation is too slow for that.
    if extrapolate:
      valid = np.all(np.isfinite(ret), axis=0)
      if not np.all(valid):
        points = tuple([q[valid] for q in query_coords[::-1]])  # xy[z]
        intp = _interpolate_points(
            points,
            query_points,
            *[c[valid] for c in ret],
            method='nearest',
        )
        for i, update in enumerate(intp):
          ret[i, ...] = update.reshape(s[1:])

    return ret

  if dim == 2:
    for z in range(coord_map.shape[1]):
      curr_map = coord_map[:, z, ...]
      rets.append(_process_map(curr_map))
    return np.stack(rets, axis=1)
  else:
    return _process_map(coord_map)


def outer_box(
    coord_map: np.ndarray,
    box: bounding_box.BoundingBox,
    stride: StrideZYX,
    target_len: StrideZYX | None = None,
) -> bounding_box.BoundingBox:
  """Returns a bounding box covering all target nodes.

  Args:
    coord_map: [2 or 3, z, y, x] coordinate map in relative format
    box: bounding box from which the coordinate map was extracted
    stride: distance between nearest neighbors of the coordinate map ([z]yx
      sequence or a single float)
    target_len: distance between nearest neighbors in the output map (defaults
      to stride)

  Returns:
    bounding box containing all (u, v,[ w]) coordinates referenced by
    the input map (x, y[, z]) -> (u, v[, w]); the bounding box is for a
    coordinate map with `target_len` node spacing
  """
  abs_map = to_absolute(coord_map, stride, box)
  extents_xyz = [(np.nanmin(c), np.nanmax(c)) for c in abs_map]

  dim = coord_map.shape[0]
  target_len_xyz = _as_vec(
      target_len if target_len is not None else stride, dim
  )[::-1]
  start = box.start.copy()
  size = box.size.copy()
  for i, ((x_min, x_max), tl) in enumerate(zip(extents_xyz, target_len_xyz)):
    x_min = int(x_min) // tl
    start[i] = x_min
    size[i] = -(int(-x_max) // tl) - x_min + 1

  return bounding_box.BoundingBox(start, size)


def inner_box(
    coord_map: np.ndarray, box: bounding_box.BoundingBox, stride: StrideZYX
) -> bounding_box.BoundingBox:
  """Returns a box within which all nodes are mapped to by coord map.

  Args:
    coord_map: [2 or 3, z, y, x] coordinate map in relative format
    box: bounding box from which the coordinate map was extracted
    stride: distance between nearest neighbors of the coordinate map

  Returns:
    bounding box, all (u, v[, w]) points contained within which have
    an entry in the (x, y[, z]) -> (u, v[, w]) map
  """
  dim = coord_map.shape[0]
  assert dim in (2, 3)
  stride = _as_vec(stride, dim)

  # Part of the map might be invalid, in which case we extrapolate
  # in order to get a fully valid array.
  int_map = to_absolute(fill_missing(coord_map, extrapolate=True), stride, box)
  x0 = np.max(np.min(int_map[0, ...], axis=-1))
  x1 = np.min(np.max(int_map[0, ...], axis=-1))
  y0 = np.max(np.min(int_map[1, ...], axis=-2))
  y1 = np.min(np.max(int_map[1, ...], axis=-2))

  x0 = int(-(-x0 // stride[-1]))
  y0 = int(-(-y0 // stride[-2]))
  x1 = x1 // stride[-1]
  y1 = y1 // stride[-2]

  if dim == 2:
    return bounding_box.BoundingBox(
        start=(x0, y0, box.start[2]),
        size=(x1 - x0 + 1, y1 - y0 + 1, box.size[2]),
    )

  z0 = np.max(np.min(int_map[2, ...], axis=-3))
  z1 = np.min(np.max(int_map[2, ...], axis=-3))
  z0 = int(-(-z0 // stride[0]))
  z1 = z1 // stride[0]

  return bounding_box.BoundingBox(
      start=(x0, y0, z0), size=(x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
  )


def invert_map(
    coord_map: np.ndarray,
    src_box: bounding_box.BoundingBox,
    dst_box: bounding_box.BoundingBox,
    stride: StrideZYX,
) -> np.ndarray:
  """Inverts a coordinate map.

  Given a (x, y[, z]) -> (u, v[, w]) map, returns a (u, v[, w]) -> (x, y[, z])
  map.

  Args:
    coord_map: [2 or 3, z, y, x] coordinate map in relative format
    src_box: box corresponding to coord_map
    dst_box: uv coordinate box for which to compute output
    stride: distance between nearest neighbors of the coordinate map ([z]yx
      sequence or a single float)

  Returns:
    inverted coordinate map in relative format
  """
  # Switch to a coordinate system originating at the first target node
  # of the coordinate map.
  coord_map = coord_map.astype(np.float64)
  dim = coord_map.shape[0]
  stride = _as_vec(stride, dim)
  src_box = src_box.adjusted_by(start=-dst_box.start, end=-dst_box.start)
  dst_box = dst_box.adjusted_by(start=-dst_box.start, end=-dst_box.start)
  coord_map = to_absolute(coord_map, stride, src_box)

  def _sel_size(box):
    if dim == 2:
      return np.mgrid[: box.size[1], : box.size[0]]
    elif dim == 3:
      return np.mgrid[: box.size[2], : box.size[1], : box.size[0]]
    else:
      raise NotImplementedError()

  src_coords = _sel_size(src_box)  # [z]yx
  for i, src in enumerate(src_coords):
    src_coords[i] = (src + src_box.start[dim - i - 1]) * stride[i]

  # ([w, ]v, u) points at which the map will be evaluated.
  query_coords = _sel_size(dst_box)
  for i, query in enumerate(query_coords):
    query_coords[i] = (query + dst_box.start[dim - i - 1]) * stride[i]

  query_points = tuple([q.ravel() for q in query_coords[::-1]])  # uv[w]

  if dim == 2:
    ret_uv = np.full(
        (2, coord_map.shape[1], dst_box.size[1], dst_box.size[0]),
        np.nan,
        dtype=coord_map.dtype,
    )

    for z in range(coord_map.shape[1]):
      valid = np.all(np.isfinite(coord_map[:, z, ...]), axis=0)
      if not np.any(valid):
        continue

      src_points = tuple([c[z][valid] for c in coord_map])
      try:
        u, v = _interpolate_points(
            src_points, query_points, *[s[valid] for s in src_coords[::-1]]
        )
        ret_uv[0, z, ...] = u.reshape(query_coords[1].shape)
        ret_uv[1, z, ...] = v.reshape(query_coords[0].shape)
      except spatial.qhull.QhullError:
        pass

    return to_relative(ret_uv, stride, dst_box)

  assert dim == 3

  ret = np.full(
      (3, dst_box.size[2], dst_box.size[1], dst_box.size[0]),
      np.nan,
      dtype=coord_map.dtype,
  )
  valid = np.all(np.isfinite(coord_map), axis=0)
  if not np.any(valid):
    return ret

  src_points = tuple([c[valid] for c in coord_map])
  try:
    u, v, w = _interpolate_points(
        src_points, query_points, *[s[valid] for s in src_coords[::-1]]
    )
    ret[0, ...] = u.reshape(query_coords[2].shape)
    ret[1, ...] = v.reshape(query_coords[1].shape)
    ret[2, ...] = w.reshape(query_coords[0].shape)
  except spatial.qhull.QhullError:
    pass

  return to_relative(ret, stride, dst_box)


def resample_map(
    coord_map: np.ndarray,
    src_box: bounding_box.BoundingBox,
    dst_box: bounding_box.BoundingBox,
    src_stride: float,
    dst_stride: float,
    method='linear',
) -> np.ndarray:
  """Resamples a coordinate map to a new grid.

  Args:
    coord_map: [2, z, y, x] coordinate map in relative format
    src_box: box corresponding to coord_map
    dst_box: target box for which to resample
    src_stride: distance between nearest neighbors of the source coordinate map
    dst_stride: distance between nearest neighbors of the target coordinate map
    method: interpolation scheme to use (linear, nearest, cubic)

  Returns:
    resampled coordinate map with dst_stride node separation
  """
  assert coord_map.shape[0] == 2

  src_y, src_x = np.mgrid[: src_box.size[1], : src_box.size[0]]
  src_y = (src_y + src_box.start[1]) * src_stride
  src_x = (src_x + src_box.start[0]) * src_stride

  tg_y, tg_x = np.mgrid[: dst_box.size[1], : dst_box.size[0]]
  tg_y = (tg_y + dst_box.start[1]) * dst_stride
  tg_x = (tg_x + dst_box.start[0]) * dst_stride
  tg_points = tg_x.ravel(), tg_y.ravel()

  ret = np.full(
      (2, coord_map.shape[1], dst_box.size[1], dst_box.size[0]),
      np.nan,
      dtype=coord_map.dtype,
  )
  for z in range(coord_map.shape[1]):
    valid = np.isfinite(coord_map[0, z, ...])
    if not np.any(valid):
      continue

    src_points = src_x[valid], src_y[valid]
    try:
      u, v = _interpolate_points(
          src_points,
          tg_points,  #
          coord_map[0, z, ...][valid],
          coord_map[1, z, ...][valid],
          method=method,
      )
      ret[0, z, ...] = u.reshape(tg_x.shape)
      ret[1, z, ...] = v.reshape(tg_y.shape)
    except spatial.qhull.QhullError:
      pass

  return ret


def compose_maps(
    map1: np.ndarray,
    box1: bounding_box.BoundingBox,
    stride1: float,
    map2: np.ndarray,
    box2: bounding_box.BoundingBox,
    stride2: float,
) -> np.ndarray:
  """Composes two coordinate maps.

  Invalid values in map2 are interpolated.

  Args:
    map1: [2, z, y, x] 1st coordinate map in relative format
    box1: box corresponding to map1
    stride1: distance between nearest neighbors of map1
    map2: [2, z, y, x] 2nd coordinate map in relative format
    box2: box corresponding to map2
    stride2: distance between nearest neighbors of map2

  Returns:
    coordinate map corresponding to map2(map1(x, y))
  """

  assert map1.shape[0] == 2
  assert map2.shape[0] == 2

  abs_map1 = to_absolute(map1, stride1, box1)
  abs_map2 = to_absolute(map2, stride2, box2)

  ret = np.full_like(map1, np.nan)

  src_y, src_x = np.mgrid[
      box2.start[1] : box2.end[1], box2.start[0] : box2.end[0]
  ]
  src_x = src_x * stride2
  src_y = src_y * stride2

  for z in range(map1.shape[1]):
    valid = np.all(np.isfinite(abs_map1[:, z, ...]), axis=0)
    if not np.any(valid):
      continue

    query_points = (abs_map1[0, z, ...][valid], abs_map1[1, z, ...][valid])  #

    valid_src = np.all(np.isfinite(abs_map2[:, z, ...]), axis=0)
    if not np.any(valid_src):
      continue

    src_points = src_x[valid_src], src_y[valid_src]
    try:
      u, v = _interpolate_points(
          src_points,
          query_points,
          abs_map2[0, z, ...][valid_src],
          abs_map2[1, z, ...][valid_src],
      )
      ret[0, z, ...][valid] = u
      ret[1, z, ...][valid] = v
    except spatial.qhull.QhullError:
      pass

  return to_relative(ret, stride1, box1)


# TODO(mjanusz): Automatically split computation into smaller boxes (overlapping
# as necessary) in order to improve precision of the calculations.
def compose_maps_fast(
    map1: jnp.ndarray,
    start1: Sequence[float],
    stride1: StrideZYX,
    map2: jnp.ndarray,
    start2: Sequence[float],
    stride2: StrideZYX,
    mode='nearest',
) -> jnp.ndarray:
  """Composes two cooordinate maps using JAX.

  Unlike compose_maps(), invalid value in either map are NOT interpolated.

  Args:
    map1: [2 or 3, z, y, x] 1st coordinate map in relative format
    start1: [z]yx origin coordinates for map1
    stride1: distance between nearest neighbors of map1; single value for all
      dimensions or a [z]yx tuple
    map2: [2 or 3, z, y, x] 2nd coordinate map in relative format
    start2: [z]yx origin coordinates for map2
    stride2: distance between nearest neighbors of map2; single value for all
      dimensions or a [z]yx tuple
    mode: interpolation mode, passed to map_coordinates

  Returns:
    coordinate map corresponding to map2(map1(z, y, x)), covering the area
    corresponding to map1 (with stride1)
  """
  assert map1.shape[0] == map2.shape[0]
  dim = map1.shape[0]

  stride1 = _as_vec(stride1, dim)
  stride2 = _as_vec(stride2, dim)
  start1 = jnp.asarray(start1)
  start2 = jnp.asarray(start2)
  origin = jnp.minimum(start1, start2)

  def _ref_grid(coord_map, start, stride):
    start = (start - origin)[-dim:]  # yx
    ranges = []
    for i in range(dim):
      # The arguments to arange have to be known at JIT time, so add the
      # (dynamic) 'start' offset separately.
      ranges.append(jnp.arange(0, coord_map.shape[4 - dim + i]) + start[i])
    ref = jnp.meshgrid(*ranges, indexing='ij')
    return [a * b for a, b in zip(ref, stride)]  # image coordinates

  ref1 = _ref_grid(map1, start1, stride1)
  ref2 = _ref_grid(map2, start2, stride2)

  if dim == 2:
    ret = jnp.zeros_like(map1)
    for z in range(map1.shape[1]):
      # Absolute values, in map2 coordinates.
      qx = (ref1[-1] + map1[0, z, ...]) / stride2[-1]
      qy = (ref1[-2] + map1[1, z, ...]) / stride2[-2]
      query_coords = jnp.array([qy, qx])  # [2, y, x]

      # Query data in absolute format and then immediately convert to relative.
      xx = (
          jax.scipy.ndimage.map_coordinates(
              map2[0, z, ...] + ref2[-1],
              query_coords,
              order=1,
              mode=mode,
              cval=np.nan,
          )
          - ref1[-1]
      )
      yy = (
          jax.scipy.ndimage.map_coordinates(
              map2[1, z, ...] + ref2[-2],
              query_coords,
              order=1,
              mode=mode,
              cval=np.nan,
          )
          - ref1[-2]
      )
      ret = ret.at[:, z, :, :].set(jnp.array([xx, yy]))
  else:
    qx = (ref1[-1] + map1[0, ...]) / stride2[-1]
    qy = (ref1[-2] + map1[1, ...]) / stride2[-2]
    qz = (ref1[-3] + map1[2, ...]) / stride2[-3]
    query_coords = jnp.array([qz, qy, qx])

    xx = (
        jax.scipy.ndimage.map_coordinates(
            map2[0, ...] + ref2[-1],
            query_coords,
            order=1,
            mode=mode,
            cval=np.nan,
        )
        - ref1[-1]
    )
    yy = (
        jax.scipy.ndimage.map_coordinates(
            map2[1, ...] + ref2[-2],
            query_coords,
            order=1,
            mode=mode,
            cval=np.nan,
        )
        - ref1[-2]
    )
    zz = (
        jax.scipy.ndimage.map_coordinates(
            map2[2, ...] + ref2[-3],
            query_coords,
            order=1,
            mode=mode,
            cval=np.nan,
        )
        - ref1[-3]
    )
    ret = jnp.array([xx, yy, zz])

  return ret


def mask_irregular(
    coord_map: np.ndarray,
    stride: Sequence[float],
    frac: float,
    max_frac: float | None = None,
    dilation_iters: int = 1,
) -> np.ndarray:
  """Masks stretched/folded parts of the map.

  Masked entries are replaced with nan's in-place.

  Args:
    coord_map: [2, y, x] single-section coordinate map in relative format
    stride: distance between nearest neighbors of the map
    frac: min. distance between target nearest neighbors, as a fraction of
      stride
    max_frac: max. distance between target nearest neighbors, as a fraction of
      stride; defaults to 2 - frac if not specified
    dilation_iters: number of dilations to apply to the node mask

  Returns:
    bool array (y, x), with True entries indicating masked elements of the
    input map
  """
  assert len(coord_map.shape) == 3
  assert coord_map.shape[0] == 2
  stride = np.asarray(stride)

  if max_frac is None:
    max_frac = 2 - frac

  stride_x, stride_y = stride
  diff_x = np.diff(coord_map[0, ...], axis=-1)
  diff_y = np.diff(coord_map[1, ...], axis=-2)
  diff_x = np.pad(diff_x, [[0, 0], [0, 1]], mode='constant') + stride_x
  diff_y = np.pad(diff_y, [[0, 1], [0, 0]], mode='constant') + stride_y

  bad = (diff_x < frac * stride_x) | (diff_y < frac * stride_y)
  bad |= (diff_x > max_frac * stride_x) | (diff_y > max_frac * stride_y)

  if dilation_iters > 0:
    bad = ndimage.morphology.binary_dilation(
        bad,
        ndimage.morphology.generate_binary_structure(2, 2),
        iterations=dilation_iters,
    )

  coord_map[0, ...][bad] = np.nan
  coord_map[1, ...][bad] = np.nan
  return bad


def make_affine_map(
    matrix: np.ndarray, box: bounding_box.BoundingBox, stride: StrideZYX
) -> np.ndarray:
  """Builds a coordinate map for an affine transform.

  Args:
    matrix: [3, 4] array, same format as ndimage.affine_transform
    box: bounding box for which to generate the map
    stride: zyx stride with which to generate the map

  Returns:
    coordinate map representing the specified affine transform
  """
  coord_map = np.array(_identity_map_absolute(box.size[::-1], stride)[::-1])
  coord_map[0, ...] += box.start[0]
  coord_map[1, ...] += box.start[1]
  coord_map[2, ...] += box.start[2]

  affine_absolute = (
      np.dot(matrix[:3, :3], coord_map.reshape((3, -1)))
      + matrix[:, 3][:, np.newaxis]
  ).reshape(coord_map.shape)
  return affine_absolute - coord_map
