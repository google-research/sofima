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
import functools as ft
import itertools
import operator
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
  data_points = np.array(data_points).T  # pyrefly: ignore[bad-assignment]
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

  assert len(value) == dim, f'Dimension mismatch: {value=} vs {dim=}'
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
  query_coords_cache: np.ndarray | None = None

  def _get_query_coords() -> np.ndarray:
    nonlocal query_coords_cache
    if query_coords_cache is None:
      if dim == 2:
        query_coords_cache = np.mgrid[: s[-2], : s[-1]]  # yx
      elif dim == 3:
        query_coords_cache = np.mgrid[: s[-3], : s[-2], : s[-1]]  # zyx
      else:
        raise ValueError(f'Unsupported coordinate map dim: {dim}')
    return query_coords_cache

  def _process_map(curr_map: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(curr_map), axis=0)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == valid.size:
      return curr_map.copy()
    if n_valid == 0:
      return np.zeros_like(curr_map) if invalid_to_zero else curr_map.copy()

    # Fast path when valid entries form a solid axis-aligned box (the common
    # case after invert_map / resample_map with margin context).
    if valid.ndim == 2:
      vy = np.any(valid, axis=1)
      vx = np.any(valid, axis=0)
      y0 = int(np.argmax(vy))
      y1 = int(valid.shape[0] - 1 - np.argmax(vy[::-1]))
      x0 = int(np.argmax(vx))
      x1 = int(valid.shape[1] - 1 - np.argmax(vx[::-1]))
      if n_valid == (y1 - y0 + 1) * (x1 - x0 + 1):
        if not extrapolate:
          return curr_map.copy()
        iy = np.clip(np.arange(valid.shape[0]), y0, y1)
        ix = np.clip(np.arange(valid.shape[1]), x0, x1)
        return curr_map[:, iy, :][:, :, ix].copy()
    elif valid.ndim == 3:
      vz = np.any(valid, axis=(1, 2))
      vy = np.any(valid, axis=(0, 2))
      vx = np.any(valid, axis=(0, 1))
      z0 = int(np.argmax(vz))
      z1 = int(valid.shape[0] - 1 - np.argmax(vz[::-1]))
      y0 = int(np.argmax(vy))
      y1 = int(valid.shape[1] - 1 - np.argmax(vy[::-1]))
      x0 = int(np.argmax(vx))
      x1 = int(valid.shape[2] - 1 - np.argmax(vx[::-1]))
      if n_valid == (z1 - z0 + 1) * (y1 - y0 + 1) * (x1 - x0 + 1):
        if not extrapolate:
          return curr_map.copy()
        iz = np.clip(np.arange(valid.shape[0]), z0, z1)
        iy = np.clip(np.arange(valid.shape[1]), y0, y1)
        ix = np.clip(np.arange(valid.shape[2]), x0, x1)
        return curr_map[:, iz, :, :][:, :, iy, :][:, :, :, ix].copy()

    ret = curr_map.copy()
    missing = ~valid

    if interpolate_first:
      q_coords = _get_query_coords()
      points = tuple(q[valid] for q in q_coords[::-1])  # xy[z]
      missing_points = tuple(q[missing] for q in q_coords[::-1])  # xy[z]
      try:
        intp = _interpolate_points(
            points, missing_points, *[c[valid] for c in curr_map]
        )
        for i, update in enumerate(intp):
          ret[i, missing] = update
      except spatial.qhull.QhullError:
        pass

    # Exact Euclidean grid nearest-neighbor fill via EDT on remaining NaNs.
    if extrapolate:
      missing_after = ~np.all(np.isfinite(ret), axis=0)
      if np.any(missing_after):
        idx = ndimage.distance_transform_edt(
            missing_after, return_distances=False, return_indices=True
        )
        assert isinstance(idx, np.ndarray)
        sel = tuple(ind[missing_after] for ind in idx)
        ret[(slice(None), missing_after)] = ret[(slice(None),) + sel]

    return ret

  if dim == 2:
    bad_z = np.flatnonzero(~np.all(np.isfinite(coord_map), axis=(0, 2, 3)))
    if bad_z.size == 0:
      return coord_map.copy()
    ret = coord_map.copy()
    for z in bad_z:
      ret[:, int(z), ...] = _process_map(coord_map[:, int(z), ...])
    return ret
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

  return bounding_box.BoundingBox(start, size)  # pyrefly: ignore[bad-argument-type]


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


@ft.partial(jax.jit, static_argnames=('hd', 'wd'))
def _invert_2d_newton_jit(
    u_rel: jax.Array,
    v_rel: jax.Array,
    off_x: float,
    off_y: float,
    hd: int,
    wd: int,
) -> tuple[jax.Array, jax.Array]:
  """Runs 6 float32 clamped bilinear Newton steps for 2D map inversion."""
  hs, ws = u_rel.shape
  max_x = float(ws - 1)
  max_y = float(hs - 1)
  qx = jnp.arange(wd, dtype=jnp.float32)[None, :]
  qy = jnp.arange(hd, dtype=jnp.float32)[:, None]
  ix = jnp.clip(
      jnp.broadcast_to(qx - off_x - jnp.mean(u_rel), (hd, wd)), 0.0, max_x
  )
  iy = jnp.clip(
      jnp.broadcast_to(qy - off_y - jnp.mean(v_rel), (hd, wd)), 0.0, max_y
  )
  ix0_w = jnp.clip(jnp.rint(ix).astype(jnp.int32), 0, ws - 1)
  iy0_w = jnp.clip(jnp.rint(iy).astype(jnp.int32), 0, hs - 1)
  ix = jnp.clip(qx - off_x - u_rel[iy0_w, ix0_w], 0.0, max_x)
  iy = jnp.clip(qy - off_y - v_rel[iy0_w, ix0_w], 0.0, max_y)

  def _step(carry, _):
    cx, cy = carry
    kx = jnp.clip(jnp.floor(cx).astype(jnp.int32), 0, ws - 2)
    ky = jnp.clip(jnp.floor(cy).astype(jnp.int32), 0, hs - 2)
    tx = cx - kx.astype(jnp.float32)
    ty = cy - ky.astype(jnp.float32)
    omx = 1.0 - tx
    omy = 1.0 - ty

    u00 = u_rel[ky, kx]
    u10 = u_rel[ky, kx + 1]
    u01 = u_rel[ky + 1, kx]
    u11 = u_rel[ky + 1, kx + 1]
    v00 = v_rel[ky, kx]
    v10 = v_rel[ky, kx + 1]
    v01 = v_rel[ky + 1, kx]
    v11 = v_rel[ky + 1, kx + 1]

    du = omy * (omx * u00 + tx * u10) + ty * (omx * u01 + tx * u11)
    dv = omy * (omx * v00 + tx * v10) + ty * (omx * v01 + tx * v11)
    ru = (cx + off_x + du) - qx
    rv = (cy + off_y + dv) - qy

    jxx = 1.0 + omy * (u10 - u00) + ty * (u11 - u01)
    jxy = omx * (u01 - u00) + tx * (u11 - u10)
    jyx = omy * (v10 - v00) + ty * (v11 - v01)
    jyy = 1.0 + omx * (v01 - v00) + tx * (v11 - v10)
    det = jxx * jyy - jxy * jyx
    inv_det = jnp.where(jnp.abs(det) > 1e-6, 1.0 / det, 0.0)
    step_x = (jyy * ru - jxy * rv) * inv_det
    step_y = (-jyx * ru + jxx * rv) * inv_det
    nx = jnp.clip(cx - step_x, 0.0, max_x)
    ny = jnp.clip(cy - step_y, 0.0, max_y)
    return (nx, ny), None

  (ix, iy), _ = jax.lax.scan(_step, (ix, iy), None, length=6)
  return ix, iy


def _invert_2d_dense_newton(
    dx_s: np.ndarray,
    dy_s: np.ndarray,
    off_x: float,
    off_y: float,
    sx: float,
    sy: float,
    hd: int,
    wd: int,
) -> tuple[np.ndarray, np.ndarray] | None:
  """Inverts a dense 2D relative map slice via Newton-Raphson if injective."""
  hs, ws = dx_s.shape
  if hs < 2 or ws < 2 or hd < 1 or wd < 1:
    return None
  if not (np.all(np.isfinite(dx_s)) and np.all(np.isfinite(dy_s))):
    return None

  u_rel = dx_s / sx
  v_rel = dy_s / sy
  ux0 = u_rel[:-1, 1:] - u_rel[:-1, :-1]
  ux1 = u_rel[1:, 1:] - u_rel[1:, :-1]
  uy0 = u_rel[1:, :-1] - u_rel[:-1, :-1]
  uy1 = u_rel[1:, 1:] - u_rel[:-1, 1:]
  vx0 = v_rel[:-1, 1:] - v_rel[:-1, :-1]
  vx1 = v_rel[1:, 1:] - v_rel[1:, :-1]
  vy0 = v_rel[1:, :-1] - v_rel[:-1, :-1]
  vy1 = v_rel[1:, 1:] - v_rel[:-1, 1:]

  # Exact Gale-Nikaido P-matrix check across all 4 cell corners.
  jxx0, jxx1 = 1.0 + ux0, 1.0 + ux1
  jyy0, jyy1 = 1.0 + vy0, 1.0 + vy1
  if (
      np.min(jxx0) <= 0.05
      or np.min(jxx1) <= 0.05
      or np.min(jyy0) <= 0.05
      or np.min(jyy1) <= 0.05
      or np.min(jxx0 * jyy0 - uy0 * vx0) <= 0.05
      or np.min(jxx0 * jyy1 - uy1 * vx0) <= 0.05
      or np.min(jxx1 * jyy0 - uy0 * vx1) <= 0.05
      or np.min(jxx1 * jyy1 - uy1 * vx1) <= 0.05
  ):
    return None

  max_x = float(ws - 1)
  max_y = float(hs - 1)
  qx = np.arange(wd, dtype=np.float64)[None, :]
  qy = np.arange(hd, dtype=np.float64)[:, None]

  if hd * wd >= 16384:
    ix_dev, iy_dev = _invert_2d_newton_jit(
        jnp.asarray(u_rel, dtype=jnp.float32),
        jnp.asarray(v_rel, dtype=jnp.float32),
        off_x,
        off_y,
        hd,
        wd,
    )
    ix = np.clip(np.asarray(ix_dev, dtype=np.float64), 0.0, max_x)
    iy = np.clip(np.asarray(iy_dev, dtype=np.float64), 0.0, max_y)
    n_iters = 4
  else:
    ix = np.clip(
        np.broadcast_to(qx - off_x - float(np.mean(u_rel)), (hd, wd)),
        0.0,
        max_x,
    )
    iy = np.clip(
        np.broadcast_to(qy - off_y - float(np.mean(v_rel)), (hd, wd)),
        0.0,
        max_y,
    )
    ix0_w = np.clip(np.rint(ix).astype(np.intp), 0, ws - 1)
    iy0_w = np.clip(np.rint(iy).astype(np.intp), 0, hs - 1)
    ix = np.clip(qx - off_x - u_rel[iy0_w, ix0_w], 0.0, max_x)
    iy = np.clip(qy - off_y - v_rel[iy0_w, ix0_w], 0.0, max_y)
    n_iters = 8

  def _eval_2d(
      cx: np.ndarray, cy: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kx = np.clip(np.floor(cx).astype(np.intp), 0, ws - 2)
    ky = np.clip(np.floor(cy).astype(np.intp), 0, hs - 2)
    tx = cx - kx
    ty = cy - ky
    omx = 1.0 - tx
    omy = 1.0 - ty

    u00 = u_rel[ky, kx]
    u10 = u_rel[ky, kx + 1]
    u01 = u_rel[ky + 1, kx]
    u11 = u_rel[ky + 1, kx + 1]
    v00 = v_rel[ky, kx]
    v10 = v_rel[ky, kx + 1]
    v01 = v_rel[ky + 1, kx]
    v11 = v_rel[ky + 1, kx + 1]

    du = omy * (omx * u00 + tx * u10) + ty * (omx * u01 + tx * u11)
    dv = omy * (omx * v00 + tx * v10) + ty * (omx * v01 + tx * v11)
    ru = (cx + off_x + du) - qx
    rv = (cy + off_y + dv) - qy

    jxx = 1.0 + omy * (u10 - u00) + ty * (u11 - u01)
    jxy = omx * (u01 - u00) + tx * (u11 - u10)
    jyx = omy * (v10 - v00) + ty * (v11 - v01)
    jyy = 1.0 + omx * (v01 - v00) + tx * (v11 - v10)
    det = jxx * jyy - jxy * jyx
    safe_det = np.where(det != 0, det, 1.0)
    inv_det = np.where(np.abs(det) > 1e-12, 1.0 / safe_det, 0.0)
    nx_raw = cx - (jyy * ru - jxy * rv) * inv_det
    ny_raw = cy - (-jyx * ru + jxx * rv) * inv_det
    return ru, rv, nx_raw, ny_raw

  for _ in range(n_iters):
    _, _, nx_raw, ny_raw = _eval_2d(ix, iy)
    nx = np.clip(nx_raw, 0.0, max_x)
    ny = np.clip(ny_raw, 0.0, max_y)
    max_delta = max(
        float(np.max(np.abs(nx - ix))), float(np.max(np.abs(ny - iy)))
    )
    ix, iy = nx, ny
    if max_delta < 1e-12:
      break

  ru, rv, nx_raw, ny_raw = _eval_2d(ix, iy)
  res = np.maximum(np.abs(ru), np.abs(rv))
  interior = (ix > 0.0) & (ix < max_x) & (iy > 0.0) & (iy < max_y)
  inward = (nx_raw > 0.0) & (nx_raw < max_x) & (ny_raw > 0.0) & (ny_raw < max_y)
  if np.any((interior | inward) & (res > 1e-8)):
    return None

  out_u = np.where(res <= 1e-8, ((ix + off_x) - qx) * sx, np.nan)
  out_v = np.where(res <= 1e-8, ((iy + off_y) - qy) * sy, np.nan)
  return out_u, out_v


def _invert_3d_dense_newton(
    rel_map: np.ndarray,
    off_xyz: tuple[float, float, float],
    stride_zyx: Sequence[float],
    dst_zyx: tuple[int, int, int],
) -> np.ndarray | None:
  """Inverts a dense 3D relative map via trilinear Newton-Raphson."""
  _, zs, hs, ws = rel_map.shape
  zd, hd, wd = dst_zyx
  if zs < 2 or hs < 2 or ws < 2 or zd < 1 or hd < 1 or wd < 1:
    return None
  if not np.all(np.isfinite(rel_map)):
    return None

  sz, sy, sx = float(stride_zyx[0]), float(stride_zyx[1]), float(stride_zyx[2])
  off_x, off_y, off_z = off_xyz
  u_rel = rel_map[0] / sx
  v_rel = rel_map[1] / sy
  w_rel = rel_map[2] / sz

  # Verify P-matrix positive principal minors across all 8 cell corners.
  for cz in (slice(0, zs - 1), slice(1, zs)):
    for cy in (slice(0, hs - 1), slice(1, hs)):
      ux = u_rel[cz, cy, 1:] - u_rel[cz, cy, :-1]
      vx = v_rel[cz, cy, 1:] - v_rel[cz, cy, :-1]
      wx = w_rel[cz, cy, 1:] - w_rel[cz, cy, :-1]
      for cx in (slice(0, ws - 1), slice(1, ws)):
        uy = u_rel[cz, 1:, cx] - u_rel[cz, :-1, cx]
        vy = v_rel[cz, 1:, cx] - v_rel[cz, :-1, cx]
        wy = w_rel[cz, 1:, cx] - w_rel[cz, :-1, cx]
        uz = u_rel[1:, cy, cx] - u_rel[:-1, cy, cx]
        vz = v_rel[1:, cy, cx] - v_rel[:-1, cy, cx]
        wz = w_rel[1:, cy, cx] - w_rel[:-1, cy, cx]
        jxx = 1.0 + ux
        jyy = 1.0 + vy
        jzz = 1.0 + wz
        m_xy = jxx * jyy - uy * vx
        m_xz = jxx * jzz - uz * wx
        m_yz = jyy * jzz - vz * wy
        det3 = (
            jxx * (jyy * jzz - vz * wy)
            - uy * (vx * jzz - vz * wx)
            + uz * (vx * wy - jyy * wx)
        )
        if (
            np.min(jxx) <= 0.05
            or np.min(jyy) <= 0.05
            or np.min(jzz) <= 0.05
            or np.min(m_xy) <= 0.05
            or np.min(m_xz) <= 0.05
            or np.min(m_yz) <= 0.05
            or np.min(det3) <= 0.05
        ):
          return None

  max_x, max_y, max_z = float(ws - 1), float(hs - 1), float(zs - 1)
  qx = np.arange(wd, dtype=np.float64)[None, None, :]
  qy = np.arange(hd, dtype=np.float64)[None, :, None]
  qz = np.arange(zd, dtype=np.float64)[:, None, None]

  ix = np.clip(
      np.broadcast_to(qx - off_x - float(np.mean(u_rel)), (zd, hd, wd)),
      0.0,
      max_x,
  )
  iy = np.clip(
      np.broadcast_to(qy - off_y - float(np.mean(v_rel)), (zd, hd, wd)),
      0.0,
      max_y,
  )
  iz = np.clip(
      np.broadcast_to(qz - off_z - float(np.mean(w_rel)), (zd, hd, wd)),
      0.0,
      max_z,
  )
  kx0 = np.clip(np.rint(ix).astype(np.intp), 0, ws - 1)
  ky0 = np.clip(np.rint(iy).astype(np.intp), 0, hs - 1)
  kz0 = np.clip(np.rint(iz).astype(np.intp), 0, zs - 1)
  ix = np.clip(qx - off_x - u_rel[kz0, ky0, kx0], 0.0, max_x)
  iy = np.clip(qy - off_y - v_rel[kz0, ky0, kx0], 0.0, max_y)
  iz = np.clip(qz - off_z - w_rel[kz0, ky0, kx0], 0.0, max_z)

  def _eval_trilin(arr, kz, ky, kx, tz, ty, tx):
    omz, omy, omx = 1.0 - tz, 1.0 - ty, 1.0 - tx
    c000 = arr[kz, ky, kx]
    c001 = arr[kz, ky, kx + 1]
    c010 = arr[kz, ky + 1, kx]
    c011 = arr[kz, ky + 1, kx + 1]
    c100 = arr[kz + 1, ky, kx]
    c101 = arr[kz + 1, ky, kx + 1]
    c110 = arr[kz + 1, ky + 1, kx]
    c111 = arr[kz + 1, ky + 1, kx + 1]
    c00 = omx * c000 + tx * c001
    c01 = omx * c010 + tx * c011
    c10 = omx * c100 + tx * c101
    c11 = omx * c110 + tx * c111
    c0 = omy * c00 + ty * c01
    c1 = omy * c10 + ty * c11
    val = omz * c0 + tz * c1
    dx = omz * (omy * (c001 - c000) + ty * (c011 - c010)) + tz * (
        omy * (c101 - c100) + ty * (c111 - c110)
    )
    dy = omz * (c01 - c00) + tz * (c11 - c10)
    dz = c1 - c0
    return val, dx, dy, dz

  def _step_3d(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray) -> tuple[
      np.ndarray,
      np.ndarray,
      np.ndarray,
      np.ndarray,
      np.ndarray,
      np.ndarray,
  ]:
    kx = np.clip(np.floor(cx).astype(np.intp), 0, ws - 2)
    ky = np.clip(np.floor(cy).astype(np.intp), 0, hs - 2)
    kz = np.clip(np.floor(cz).astype(np.intp), 0, zs - 2)
    tx, ty, tz = cx - kx, cy - ky, cz - kz
    du, ux, uy, uz = _eval_trilin(u_rel, kz, ky, kx, tz, ty, tx)
    dv, vx, vy, vz = _eval_trilin(v_rel, kz, ky, kx, tz, ty, tx)
    dw, wx, wy, wz = _eval_trilin(w_rel, kz, ky, kx, tz, ty, tx)
    ru = (cx + off_x + du) - qx
    rv = (cy + off_y + dv) - qy
    rw = (cz + off_z + dw) - qz
    jxx, jxy, jxz = 1.0 + ux, uy, uz
    jyx, jyy, jyz = vx, 1.0 + vy, vz
    jzx, jzy, jzz = wx, wy, 1.0 + wz
    c00 = jyy * jzz - jyz * jzy
    c01 = -(jyx * jzz - jyz * jzx)
    c02 = jyx * jzy - jyy * jzx
    c10 = -(jxy * jzz - jxz * jzy)
    c11 = jxx * jzz - jxz * jzx
    c12 = -(jxx * jzy - jxy * jzx)
    c20 = jxy * jyz - jxz * jyy
    c21 = -(jxx * jyz - jxz * jyx)
    c22 = jxx * jyy - jxy * jyx
    det = jxx * c00 + jxy * c01 + jxz * c02
    inv_det = np.where(
        np.abs(det) > 1e-12, 1.0 / np.where(det != 0, det, 1.0), 0.0
    )
    nx_raw = cx - (c00 * ru + c10 * rv + c20 * rw) * inv_det
    ny_raw = cy - (c01 * ru + c11 * rv + c21 * rw) * inv_det
    nz_raw = cz - (c02 * ru + c12 * rv + c22 * rw) * inv_det
    return ru, rv, rw, nx_raw, ny_raw, nz_raw

  for _ in range(9):
    _, _, _, nx_raw, ny_raw, nz_raw = _step_3d(ix, iy, iz)
    nx = np.clip(nx_raw, 0.0, max_x)
    ny = np.clip(ny_raw, 0.0, max_y)
    nz = np.clip(nz_raw, 0.0, max_z)
    max_delta = max(
        float(np.max(np.abs(nx - ix))),
        float(np.max(np.abs(ny - iy))),
        float(np.max(np.abs(nz - iz))),
    )
    ix, iy, iz = nx, ny, nz
    if max_delta < 1e-12:
      break

  ru, rv, rw, nx_raw, ny_raw, nz_raw = _step_3d(ix, iy, iz)
  res = np.maximum(np.maximum(np.abs(ru), np.abs(rv)), np.abs(rw))
  interior = (
      (ix > 0.0)
      & (ix < max_x)
      & (iy > 0.0)
      & (iy < max_y)
      & (iz > 0.0)
      & (iz < max_z)
  )
  inward = (
      (nx_raw > 0.0)
      & (nx_raw < max_x)
      & (ny_raw > 0.0)
      & (ny_raw < max_y)
      & (nz_raw > 0.0)
      & (nz_raw < max_z)
  )
  if np.any((interior | inward) & (res > 1e-8)):
    return None

  ret = np.full((3, zd, hd, wd), np.nan, dtype=np.float64)
  valid_out = res <= 1e-8
  ret[0] = np.where(valid_out, ((ix + off_x) - qx) * sx, np.nan)
  ret[1] = np.where(valid_out, ((iy + off_y) - qy) * sy, np.nan)
  ret[2] = np.where(valid_out, ((iz + off_z) - qz) * sz, np.nan)
  return ret


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
  rel_map = coord_map.astype(np.float64)
  dim = rel_map.shape[0]
  stride = _as_vec(stride, dim)
  off_xyz = (
      float(src_box.start[0] - dst_box.start[0]),
      float(src_box.start[1] - dst_box.start[1]),
      float(src_box.start[2] - dst_box.start[2]),
  )

  if dim == 3:
    fast_3d = _invert_3d_dense_newton(
        rel_map,
        off_xyz,
        stride,
        (int(dst_box.size[2]), int(dst_box.size[1]), int(dst_box.size[0])),
    )
    if fast_3d is not None:
      return fast_3d

  if dim == 2:
    hd, wd = int(dst_box.size[1]), int(dst_box.size[0])
    sy, sx = float(stride[0]), float(stride[1])
    ret_fast = np.full((2, rel_map.shape[1], hd, wd), np.nan, dtype=np.float64)
    fallback_zs: list[int] = []
    for z in range(rel_map.shape[1]):
      if not np.any(np.isfinite(rel_map[:, z, ...])):
        continue
      fast_z = _invert_2d_dense_newton(
          rel_map[0, z], rel_map[1, z], off_xyz[0], off_xyz[1], sx, sy, hd, wd
      )
      if fast_z is not None:
        ret_fast[0, z], ret_fast[1, z] = fast_z
      else:
        fallback_zs.append(z)
    if not fallback_zs:
      return ret_fast
  else:
    ret_fast = None
    fallback_zs = []

  # Switch to a coordinate system originating at the first target node
  # of the coordinate map for Delaunay fallback.
  src_box = src_box.adjusted_by(start=-dst_box.start, end=-dst_box.start)  # pyrefly: ignore[bad-argument-type]
  dst_box = dst_box.adjusted_by(start=-dst_box.start, end=-dst_box.start)  # pyrefly: ignore[bad-argument-type]
  abs_map = to_absolute(rel_map, stride, src_box)

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
    assert ret_fast is not None
    ret_uv = np.full(
        (2, len(fallback_zs), dst_box.size[1], dst_box.size[0]),
        np.nan,
        dtype=abs_map.dtype,
    )
    for out_i, z in enumerate(fallback_zs):
      valid = np.all(np.isfinite(abs_map[:, z, ...]), axis=0)
      if not np.any(valid):
        continue
      src_points = tuple([c[z][valid] for c in abs_map])
      try:
        u, v = _interpolate_points(
            src_points, query_points, *[s[valid] for s in src_coords[::-1]]
        )
        ret_uv[0, out_i, ...] = u.reshape(query_coords[1].shape)
        ret_uv[1, out_i, ...] = v.reshape(query_coords[0].shape)
      except spatial.qhull.QhullError:
        pass

    rel_fb = to_relative(ret_uv, stride, dst_box)
    for out_i, z in enumerate(fallback_zs):
      ret_fast[:, z, ...] = rel_fb[:, out_i, ...]
    return ret_fast

  assert dim == 3

  ret = np.full(
      (3, dst_box.size[2], dst_box.size[1], dst_box.size[0]),
      np.nan,
      dtype=abs_map.dtype,
  )
  valid = np.all(np.isfinite(abs_map), axis=0)
  if not np.any(valid):
    return ret

  src_points = tuple([c[valid] for c in abs_map])
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


@jax.jit
def _compose_maps_dense_jit(
    m1: jax.Array,
    m2: jax.Array,
    ratio: float,
    inv_s2: float,
    off_x: float,
    off_y: float,
) -> jax.Array:
  """JIT-compiled bilinear relative coordinate map composition."""
  h1, w1 = m1.shape[2], m1.shape[3]
  h2, w2 = m2.shape[2], m2.shape[3]
  ix = jnp.arange(w1, dtype=jnp.float32)[jnp.newaxis, jnp.newaxis, :]
  iy = jnp.arange(h1, dtype=jnp.float32)[jnp.newaxis, :, jnp.newaxis]
  qx = ix * ratio + off_x + m1[0] * inv_s2
  qy = iy * ratio + off_y + m1[1] * inv_s2
  in_b = (
      jnp.isfinite(qx)
      & jnp.isfinite(qy)
      & (qx >= 0.0)
      & (qx <= float(w2 - 1))
      & (qy >= 0.0)
      & (qy <= float(h2 - 1))
  )
  qx_c = jnp.clip(jnp.where(in_b, qx, 0.0), 0.0, float(w2 - 1))
  qy_c = jnp.clip(jnp.where(in_b, qy, 0.0), 0.0, float(h2 - 1))
  kx = jnp.clip(jnp.floor(qx_c).astype(jnp.int32), 0, w2 - 2)
  ky = jnp.clip(jnp.floor(qy_c).astype(jnp.int32), 0, h2 - 2)
  wx1 = qx_c - kx.astype(jnp.float32)
  wx0 = 1.0 - wx1
  wy1 = qy_c - ky.astype(jnp.float32)
  wy0 = 1.0 - wy1

  def _sample_z(m1_z, m2_z, ky_z, kx_z, wy0_z, wy1_z, wx0_z, wx1_z, in_b_z):
    v00 = m2_z[:, ky_z, kx_z]
    v01 = m2_z[:, ky_z, kx_z + 1]
    v10 = m2_z[:, ky_z + 1, kx_z]
    v11 = m2_z[:, ky_z + 1, kx_z + 1]
    interp = wy0_z * (wx0_z * v00 + wx1_z * v01) + wy1_z * (
        wx0_z * v10 + wx1_z * v11
    )
    return jnp.where(in_b_z[jnp.newaxis, ...], m1_z + interp, jnp.nan)

  out_z = jax.vmap(_sample_z, in_axes=(1, 1, 0, 0, 0, 0, 0, 0, 0), out_axes=1)(
      m1, m2, ky, kx, wy0, wy1, wx0, wx1, in_b
  )
  return out_z


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
  ws, hs = int(src_box.size[0]), int(src_box.size[1])
  wd, hd = int(dst_box.size[0]), int(dst_box.size[1])
  nz = coord_map.shape[1]

  ret = np.full(
      (2, nz, hd, wd),
      np.nan,
      dtype=coord_map.dtype,
  )
  if nz == 0 or wd == 0 or hd == 0:
    return ret

  can_regular = ws >= 2 and hs >= 2 and method in ('linear', 'nearest')
  if can_regular:
    finite_z = np.all(np.isfinite(coord_map), axis=(0, 2, 3))
  else:
    finite_z = np.zeros(nz, dtype=bool)

  if np.any(finite_z):
    dense_idx = np.flatnonzero(finite_z)
    sub = coord_map[:, dense_idx, :, :].astype(np.float64, copy=False)
    fx = (
        (np.arange(wd, dtype=np.float64) + dst_box.start[0]) * dst_stride
    ) / src_stride - src_box.start[0]
    fy = (
        (np.arange(hd, dtype=np.float64) + dst_box.start[1]) * dst_stride
    ) / src_stride - src_box.start[1]
    if method == 'nearest':
      kx = np.clip(np.ceil(fx - 0.5).astype(np.intp), 0, ws - 1)
      ky = np.clip(np.ceil(fy - 0.5).astype(np.intp), 0, hs - 1)
      ret[:, dense_idx, :, :] = sub[:, :, ky, :][:, :, :, kx].astype(
          coord_map.dtype, copy=False
      )
    else:
      oob_x = (fx < 0.0) | (fx > ws - 1)
      oob_y = (fy < 0.0) | (fy > hs - 1)
      fx_c = np.clip(fx, 0.0, ws - 1.0)
      fy_c = np.clip(fy, 0.0, hs - 1.0)
      kx = np.clip(np.floor(fx_c).astype(np.intp), 0, ws - 2)
      ky = np.clip(np.floor(fy_c).astype(np.intp), 0, hs - 2)
      wx1 = fx_c - kx
      wx0 = 1.0 - wx1
      wy1 = fy_c - ky
      wy0 = 1.0 - wy1
      row = sub[..., kx] * wx0 + sub[..., kx + 1] * wx1
      interp = (
          row[:, :, ky, :] * wy0[:, np.newaxis]
          + row[:, :, ky + 1, :] * wy1[:, np.newaxis]
      )
      if np.any(oob_y):
        interp[:, :, oob_y, :] = np.nan
      if np.any(oob_x):
        interp[:, :, :, oob_x] = np.nan
      ret[:, dense_idx, :, :] = interp.astype(coord_map.dtype, copy=False)

    if np.all(finite_z):
      return ret

  src_y, src_x = np.mgrid[:hs, :ws]
  src_y = (src_y + src_box.start[1]) * src_stride
  src_x = (src_x + src_box.start[0]) * src_stride

  tg_y, tg_x = np.mgrid[:hd, :wd]
  tg_y = (tg_y + dst_box.start[1]) * dst_stride
  tg_x = (tg_x + dst_box.start[0]) * dst_stride
  tg_points = tg_x.ravel(), tg_y.ravel()

  for z in np.flatnonzero(~finite_z):
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
  nz, h1, w1 = map1.shape[1], map1.shape[2], map1.shape[3]
  h2, w2 = map2.shape[2], map2.shape[3]
  ret = np.full_like(map1, np.nan)
  if nz == 0 or h1 == 0 or w1 == 0:
    return ret

  assert map2.shape[1] >= nz
  if h2 >= 2 and w2 >= 2:
    dense_src_z = np.all(np.isfinite(map2[:, :nz, ...]), axis=(0, 2, 3))
  else:
    dense_src_z = np.zeros(nz, dtype=bool)

  if np.any(dense_src_z):
    dense_idx = np.flatnonzero(dense_src_z)
    ratio = float(stride1) / float(stride2)
    inv_s2 = 1.0 / float(stride2)
    off_x = (
        float(box1.start[0]) * float(stride1)
        - float(box2.start[0]) * float(stride2)
    ) * inv_s2
    off_y = (
        float(box1.start[1]) * float(stride1)
        - float(box2.start[1]) * float(stride2)
    ) * inv_s2

    if (
        map1.dtype != np.float64
        and map2.dtype != np.float64
        and dense_idx.size * h1 * w1 >= 16384
    ):
      out_dev = _compose_maps_dense_jit(
          jnp.asarray(map1[:, dense_idx, :, :], dtype=jnp.float32),
          jnp.asarray(map2[:, dense_idx, :, :], dtype=jnp.float32),
          ratio,
          inv_s2,
          off_x,
          off_y,
      )
      ret[:, dense_idx, :, :] = np.asarray(out_dev).astype(
          ret.dtype, copy=False
      )
    else:
      m1_sub = map1[:, dense_idx, :, :].astype(np.float64, copy=False)
      m2_sub = map2[:, dense_idx, :, :].astype(np.float64, copy=False)
      ix = np.arange(w1, dtype=np.float64)[np.newaxis, np.newaxis, :]
      iy = np.arange(h1, dtype=np.float64)[np.newaxis, :, np.newaxis]
      qx = ix * ratio + off_x + m1_sub[0] * inv_s2
      qy = iy * ratio + off_y + m1_sub[1] * inv_s2
      in_b = (
          np.isfinite(qx)
          & np.isfinite(qy)
          & (qx >= 0.0)
          & (qx <= w2 - 1)
          & (qy >= 0.0)
          & (qy <= h2 - 1)
      )
      qx_c = np.clip(np.where(in_b, qx, 0.0), 0.0, w2 - 1.0)
      qy_c = np.clip(np.where(in_b, qy, 0.0), 0.0, h2 - 1.0)
      kx = np.clip(np.floor(qx_c).astype(np.intp), 0, w2 - 2)
      ky = np.clip(np.floor(qy_c).astype(np.intp), 0, h2 - 2)
      wx1 = qx_c - kx
      wx0 = 1.0 - wx1
      wy1 = qy_c - ky
      wy0 = 1.0 - wy1
      zi = np.arange(dense_idx.size, dtype=np.intp)[:, np.newaxis, np.newaxis]
      v00 = m2_sub[:, zi, ky, kx]
      v01 = m2_sub[:, zi, ky, kx + 1]
      v10 = m2_sub[:, zi, ky + 1, kx]
      v11 = m2_sub[:, zi, ky + 1, kx + 1]
      composed = m1_sub + (
          wy0[np.newaxis, ...]
          * (wx0[np.newaxis, ...] * v00 + wx1[np.newaxis, ...] * v01)
          + wy1[np.newaxis, ...]
          * (wx0[np.newaxis, ...] * v10 + wx1[np.newaxis, ...] * v11)
      )
      composed[:, ~in_b] = np.nan
      ret[:, dense_idx, :, :] = composed.astype(ret.dtype, copy=False)

    if np.all(dense_src_z):
      return ret

  sparse_idx = np.flatnonzero(~dense_src_z)
  abs_map1 = to_absolute(
      map1[:, sparse_idx, :, :].astype(np.float64, copy=False),
      stride1,
      box1,
  )
  abs_map2 = to_absolute(
      map2[:, sparse_idx, :, :].astype(np.float64, copy=False),
      stride2,
      box2,
  )

  src_y, src_x = np.mgrid[
      box2.start[1] : box2.end[1], box2.start[0] : box2.end[0]
  ]
  src_x = src_x.astype(np.float64) * stride2
  src_y = src_y.astype(np.float64) * stride2

  sparse_ret = np.full_like(abs_map1, np.nan)
  for loc_z in range(sparse_idx.size):
    valid = np.all(np.isfinite(abs_map1[:, loc_z, ...]), axis=0)
    if not np.any(valid):
      continue

    query_points = (
        abs_map1[0, loc_z, ...][valid],
        abs_map1[1, loc_z, ...][valid],
    )

    valid_src = np.all(np.isfinite(abs_map2[:, loc_z, ...]), axis=0)
    if not np.any(valid_src):
      continue

    src_points = src_x[valid_src], src_y[valid_src]
    try:
      u, v = _interpolate_points(
          src_points,
          query_points,
          abs_map2[0, loc_z, ...][valid_src],
          abs_map2[1, loc_z, ...][valid_src],
      )
      sparse_ret[0, loc_z, ...][valid] = u
      sparse_ret[1, loc_z, ...][valid] = v
    except spatial.qhull.QhullError:
      pass

  ret[:, sparse_idx, :, :] = to_relative(
      sparse_ret,
      stride1,
      box1,
  ).astype(ret.dtype, copy=False)
  return ret


def _mirror_index_fixer(index: jnp.ndarray, size: int) -> jnp.ndarray:
  s = size - 1
  if s == 0:
    return jnp.zeros_like(index)
  return jnp.abs((index + s) % (2 * s) - s)


def _reflect_index_fixer(index: jnp.ndarray, size: int) -> jnp.ndarray:
  return jnp.floor_divide(
      _mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2
  )


def _multichannel_linear_interp(
    inputs: jnp.ndarray,
    coordinates: Sequence[jnp.ndarray],
    mode: str,
    cval: float,
) -> jnp.ndarray:
  """Interpolates all leading channels with shared multilinear weights."""
  spatial_shape = inputs.shape[1:]
  cval_arr = jnp.asarray(cval, dtype=inputs.dtype)

  interp_axes = []
  for coord, size in zip(coordinates, spatial_shape):
    lower = jnp.floor(coord)
    upper_w = coord - lower
    lower_w = 1.0 - upper_w
    idx0 = lower.astype(jnp.int32)
    axes_nodes = []
    for idx, weight in ((idx0, lower_w), (idx0 + 1, upper_w)):
      valid = None
      if mode == 'nearest':
        fixed = jnp.clip(idx, 0, size - 1)
      elif mode == 'constant':
        fixed = jnp.clip(idx, 0, size - 1)
        valid = (idx >= 0) & (idx < size)
      elif mode == 'wrap':
        fixed = idx % size
      elif mode == 'mirror':
        fixed = _mirror_index_fixer(idx, size)
      elif mode == 'reflect':
        fixed = _reflect_index_fixer(idx, size)
      else:
        raise NotImplementedError(f'Unsupported interpolation mode: {mode}')
      axes_nodes.append((fixed, valid, weight))
    interp_axes.append(axes_nodes)

  acc = None
  for items in itertools.product(*interp_axes):
    indices = tuple(item[0] for item in items)
    validities = [item[1] for item in items if item[1] is not None]
    weights = [item[2] for item in items]
    w_prod = ft.reduce(operator.mul, weights)
    gathered = inputs[(slice(None), *indices)]
    if validities:
      all_valid = ft.reduce(operator.and_, validities)
      gathered = jnp.where(all_valid[None, ...], gathered, cval_arr)
    term = w_prod[None, ...] * gathered
    acc = term if acc is None else acc + term
  assert acc is not None
  return acc.astype(inputs.dtype)


@ft.partial(jax.jit, static_argnames=('mode',))
def _compose_maps_fast_jit(
    map1: jnp.ndarray,
    start1: jnp.ndarray,
    stride1: jnp.ndarray,
    map2: jnp.ndarray,
    start2: jnp.ndarray,
    stride2: jnp.ndarray,
    mode: str,
) -> jnp.ndarray:
  """JIT-compiled multi-slice coordinate map composition."""
  dim = map1.shape[0]
  origin = jnp.minimum(start1, start2)

  def _ref_axes(
      coord_map: jnp.ndarray, start: jnp.ndarray, stride: jnp.ndarray
  ):
    rel_start = (start - origin)[-dim:].astype(coord_map.dtype)
    axes = []
    for i in range(dim):
      n = coord_map.shape[4 - dim + i]
      shape = [1] * dim
      shape[i] = n
      ax = (jnp.arange(n, dtype=coord_map.dtype) + rel_start[i]) * stride[i]
      axes.append(ax.reshape(shape))
    return axes

  ref1 = _ref_axes(map1, start1, stride1)
  ref2 = _ref_axes(map2, start2, stride2)
  off2 = (start2 - origin)[-dim:].astype(map1.dtype) * stride2

  if dim == 2:
    z_len = map1.shape[1]
    qx = (ref1[-1][None, :, :] + map1[0] - off2[-1]) / stride2[-1]
    qy = (ref1[-2][None, :, :] + map1[1] - off2[-2]) / stride2[-2]
    abs_map2_x = map2[0, :z_len, ...] + ref2[-1][None, :, :]
    abs_map2_y = map2[1, :z_len, ...] + ref2[-2][None, :, :]
    abs_map2 = jnp.stack([abs_map2_x, abs_map2_y], axis=1)  # [z, 2, y, x]

    def _interp_slice(m2_z, qy_z, qx_z):
      return _multichannel_linear_interp(
          m2_z, (qy_z, qx_z), mode=mode, cval=np.nan
      )

    sampled = jax.vmap(_interp_slice)(abs_map2, qy, qx)  # [z, 2, y, x]
    sampled = jnp.swapaxes(sampled, 0, 1)  # [2, z, y, x]
    return jnp.stack(
        [
            sampled[0] - ref1[-1][None, :, :],
            sampled[1] - ref1[-2][None, :, :],
        ],
        axis=0,
    )

  qx = (ref1[-1] + map1[0] - off2[-1]) / stride2[-1]
  qy = (ref1[-2] + map1[1] - off2[-2]) / stride2[-2]
  qz = (ref1[-3] + map1[2] - off2[-3]) / stride2[-3]
  abs_map2 = jnp.stack(
      [
          map2[0] + ref2[-1],
          map2[1] + ref2[-2],
          map2[2] + ref2[-3],
      ],
      axis=0,
  )
  sampled = _multichannel_linear_interp(
      abs_map2, (qz, qy, qx), mode=mode, cval=np.nan
  )
  return jnp.stack(
      [
          sampled[0] - ref1[-1],
          sampled[1] - ref1[-2],
          sampled[2] - ref1[-3],
      ],
      axis=0,
  )


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
  map1_arr = jnp.asarray(map1)
  map2_arr = jnp.asarray(map2)
  dtype = jnp.result_type(map1_arr.dtype, map2_arr.dtype, jnp.float32)
  stride1_arr = jnp.asarray(_as_vec(stride1, dim), dtype=dtype)
  stride2_arr = jnp.asarray(_as_vec(stride2, dim), dtype=dtype)
  start1_arr = jnp.asarray(start1)
  start2_arr = jnp.asarray(start2)
  return _compose_maps_fast_jit(
      map1_arr.astype(dtype),
      start1_arr,
      stride1_arr,
      map2_arr.astype(dtype),
      start2_arr,
      stride2_arr,
      mode=mode,
  )


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
  return mask_irregular_batch(
      coord_map[:, np.newaxis, ...],
      stride,
      frac,
      max_frac=max_frac,
      dilation_iters=dilation_iters,
  )[0]


def mask_irregular_batch(
    coord_map: np.ndarray,
    stride: Sequence[float],
    frac: float,
    max_frac: float | None = None,
    dilation_iters: int = 1,
) -> np.ndarray:
  """Masks stretched/folded entries across all Z slices of a [2, z, y, x] map.

  Masked entries are replaced with nan's in-place.

  Args:
    coord_map: [2, z, y, x] coordinate map in relative format
    stride: distance between nearest neighbors of the map
    frac: min. distance between target nearest neighbors, as a fraction of
      stride
    max_frac: max. distance between target nearest neighbors, as a fraction of
      stride; defaults to 2 - frac if not specified
    dilation_iters: number of dilations to apply to the node mask

  Returns:
    bool array (z, y, x), with True entries indicating masked elements of the
    input map
  """
  assert coord_map.ndim == 4
  assert coord_map.shape[0] == 2
  if max_frac is None:
    max_frac = 2 - frac

  stride_arr = np.asarray(stride)
  stride_x, stride_y = float(stride_arr[0]), float(stride_arr[1])
  diff_x = np.diff(coord_map[0], axis=-1)
  diff_y = np.diff(coord_map[1], axis=-2)
  diff_x = np.pad(diff_x, [[0, 0], [0, 0], [0, 1]], mode='constant') + stride_x
  diff_y = np.pad(diff_y, [[0, 0], [0, 1], [0, 0]], mode='constant') + stride_y

  bad = (diff_x < frac * stride_x) | (diff_y < frac * stride_y)
  bad |= (diff_x > max_frac * stride_x) | (diff_y > max_frac * stride_y)

  if dilation_iters > 0 and bad.shape[0] > 0:
    struct_3d = ndimage.generate_binary_structure(2, 2)[np.newaxis, :, :]
    bad = ndimage.binary_dilation(
        bad, structure=struct_3d, iterations=dilation_iters
    )

  coord_map[0][bad] = np.nan
  coord_map[1][bad] = np.nan
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
  coord_map = np.array(_identity_map_absolute(box.size[::-1], stride)[::-1])  # pyrefly: ignore[bad-argument-type]
  coord_map[0, ...] += box.start[0]
  coord_map[1, ...] += box.start[1]
  coord_map[2, ...] += box.start[2]

  affine_absolute = (
      np.dot(matrix[:3, :3], coord_map.reshape((3, -1)))
      + matrix[:, 3][:, np.newaxis]
  ).reshape(coord_map.shape)
  return affine_absolute - coord_map
