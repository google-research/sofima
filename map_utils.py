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
import math
import operator
from typing import Any, Sequence
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


def _boundary_valid_mask(valid: np.ndarray) -> np.ndarray:
  """Returns valid grid nodes not strictly surrounded by 3^d valid neighbors."""
  if any(s < 3 for s in valid.shape):
    return valid
  shifts = (slice(None, -2), slice(1, -1), slice(2, None))
  strictly_interior = ft.reduce(
      operator.and_,
      (valid[idx] for idx in itertools.product(shifts, repeat=valid.ndim)),
  )
  boundary = valid.copy()
  boundary[(slice(1, -1),) * valid.ndim] &= ~strictly_interior
  return boundary


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
      src_mask = _boundary_valid_mask(valid) if dim == 2 else valid
      points = tuple(q[src_mask] for q in q_coords[::-1])  # xy[z]
      missing_points = tuple(q[missing] for q in q_coords[::-1])  # xy[z]
      try:
        intp = _interpolate_points(
            points, missing_points, *[c[src_mask] for c in curr_map]
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


def _det_matrix(mat: Sequence[Sequence[Any]]) -> Any:
  """Computes the determinant of a small square matrix of arrays."""
  n = len(mat)
  if n == 1:
    return mat[0][0]
  if n == 2:
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
  acc = None
  for col in range(n):
    minor = [[row[c] for c in range(n) if c != col] for row in mat[1:]]
    term = mat[0][col] * _det_matrix(minor)
    acc = term if acc is None else (acc + term if col % 2 == 0 else acc - term)
  assert acc is not None
  return acc


def _cofactor_matrix(mat: Sequence[Sequence[Any]]) -> list[list[Any]]:
  """Computes cofactor matrix C[r][c] = (-1)^(r+c) det(minor(mat, r, c))."""
  n = len(mat)
  cof = []
  for r in range(n):
    row_cof = []
    for c in range(n):
      minor = [
          [row[cc] for cc in range(n) if cc != c]
          for rr, row in enumerate(mat)
          if rr != r
      ]
      val = _det_matrix(minor)
      row_cof.append(val if (r + c) % 2 == 0 else -val)
    cof.append(row_cof)
  return cof


def _eval_channel_val_and_grad(
    corners: list[Any],
    t_axes: Sequence[Any],
) -> tuple[Any, list[Any]]:
  """Contracts 2^d corner arrays for one channel into (val, [d/dc_0, ...])."""
  vals = corners
  derivs: list[list[Any]] = []
  for a in range(len(t_axes) - 1, -1, -1):
    t = t_axes[a]
    omt = 1.0 - t
    derivs = [
        [omt * lo + t * hi for lo, hi in zip(d[0::2], d[1::2])] for d in derivs
    ]
    derivs.insert(0, [hi - lo for lo, hi in zip(vals[0::2], vals[1::2])])
    vals = [omt * lo + t * hi for lo, hi in zip(vals[0::2], vals[1::2])]
  return vals[0], [d[0] for d in derivs]


def _lin_strides_and_offsets(
    src_spatial: tuple[int, ...],
) -> tuple[list[int], list[int]]:
  """Returns row-major linear strides and 2^d cell corner offsets."""
  dim = len(src_spatial)
  strides = [1] * dim
  for a in range(dim - 2, -1, -1):
    strides[a] = strides[a + 1] * src_spatial[a + 1]
  offsets = [
      sum(bit * strides[a] for a, bit in enumerate(bits))
      for bits in itertools.product((0, 1), repeat=dim)
  ]
  return strides, offsets


def _newton_step_flat(
    xp: Any,
    curr_c: Sequence[Any],
    u_flats: Sequence[Any],
    off_axes: Any,
    q_axes: Sequence[Any],
    src_spatial: tuple[int, ...],
    lin_strides: Sequence[int],
    corner_offsets: Sequence[int],
    idx_dtype: Any,
    val_dtype: Any,
    det_tol: float,
) -> tuple[list[Any], list[Any]]:
  """Computes residuals and unclamped Newton update via 1D flat-offset gathers."""
  dim = len(curr_c)
  k_axes = [
      xp.clip(xp.floor(curr_c[a]).astype(idx_dtype), 0, src_spatial[a] - 2)
      for a in range(dim)
  ]
  t_axes = [
      curr_c[a] - (k_axes[a].astype(val_dtype) if xp is jnp else k_axes[a])
      for a in range(dim)
  ]
  base = k_axes[0] * lin_strides[0]
  for a in range(1, dim):
    term = k_axes[a] if lin_strides[a] == 1 else k_axes[a] * lin_strides[a]
    base = base + term
  c_idxs = [base if off == 0 else base + off for off in corner_offsets]
  res_list = []
  j_mat = []
  for r in range(dim):
    du_r, grad_r = _eval_channel_val_and_grad(
        [u_flats[r][idx] for idx in c_idxs], t_axes
    )
    res_list.append((curr_c[r] + off_axes[r] + du_r) - q_axes[r])
    j_mat.append([(1.0 if r == c else 0.0) + grad_r[c] for c in range(dim)])
  cof = _cofactor_matrix(j_mat)
  det = j_mat[0][0] * cof[0][0]
  for c in range(1, dim):
    det = det + j_mat[0][c] * cof[0][c]
  safe_det = xp.where(det != 0, det, 1.0)
  inv_det = xp.where(xp.abs(det) > det_tol, 1.0 / safe_det, 0.0)
  nxt_raw = []
  for col in range(dim):
    num = cof[0][col] * res_list[0]
    for r in range(1, dim):
      num = num + cof[r][col] * res_list[r]
    nxt_raw.append(curr_c[col] - num * inv_det)
  return res_list, nxt_raw


@ft.partial(jax.jit, static_argnames=('src_spatial', 'dst_spatial'))
def _invert_newton_jit(
    u_flats: tuple[jax.Array, ...],
    off_axes: jax.Array,
    src_spatial: tuple[int, ...],
    dst_spatial: tuple[int, ...],
) -> tuple[jax.Array, ...]:
  """Runs 6 float32 clamped multilinear Newton steps for map inversion."""
  dim = len(src_spatial)
  lin_strides, corner_offsets = _lin_strides_and_offsets(src_spatial)
  q_axes = [
      jnp.arange(d, dtype=jnp.float32).reshape(
          (1,) * a + (d,) + (1,) * (dim - 1 - a)
      )
      for a, d in enumerate(dst_spatial)
  ]
  c_axes = [
      jnp.clip(
          jnp.broadcast_to(
              q_axes[a] - off_axes[a] - jnp.mean(u_flats[a]), dst_spatial
          ),
          0.0,
          float(src_spatial[a] - 1),
      )
      for a in range(dim)
  ]
  base0 = (
      jnp.clip(jnp.rint(c_axes[0]).astype(jnp.int32), 0, src_spatial[0] - 1)
      * lin_strides[0]
  )
  for a in range(1, dim):
    ka = jnp.clip(jnp.rint(c_axes[a]).astype(jnp.int32), 0, src_spatial[a] - 1)
    base0 = base0 + (ka if lin_strides[a] == 1 else ka * lin_strides[a])
  c_axes = [
      jnp.clip(
          q_axes[a] - off_axes[a] - u_flats[a][base0],
          0.0,
          float(src_spatial[a] - 1),
      )
      for a in range(dim)
  ]

  def _step(carry, _):
    _, nxt_raw = _newton_step_flat(
        jnp,
        carry,
        u_flats,
        off_axes,
        q_axes,
        src_spatial,
        lin_strides,
        corner_offsets,
        jnp.int32,
        jnp.float32,
        1e-6,
    )
    nxt = tuple(
        jnp.clip(nxt_raw[a], 0.0, float(src_spatial[a] - 1)).astype(
            jnp.float32
        )
        for a in range(dim)
    )
    return nxt, None

  final_c, _ = jax.lax.scan(_step, tuple(c_axes), None, length=6)
  return final_c


def _invert_dense_newton(
    sub_map: np.ndarray,
    off_xyz: Sequence[float],
    stride: Sequence[float],
    dst_spatial: tuple[int, ...],
) -> np.ndarray | None:
  """Inverts a dense relative map via multilinear Newton-Raphson if injective."""
  dim = sub_map.shape[0]
  src_spatial = sub_map.shape[1:]
  if any(s < 2 for s in src_spatial) or any(d < 1 for d in dst_spatial):
    return None
  if not np.all(np.isfinite(sub_map)):
    return None

  # Spatial axis order ([z, ]y, x) matches stride, src_spatial, and dst_spatial.
  u_axes = [
      np.ascontiguousarray(sub_map[dim - 1 - a] / float(stride[a]))
      for a in range(dim)
  ]
  off_axes = [float(off_xyz[dim - 1 - a]) for a in range(dim)]

  # Gale-Nikaido P-matrix check across all 2^dim cell corners.
  edge_diffs = [
      [np.diff(u_axes[r], axis=c) for c in range(dim)] for r in range(dim)
  ]
  if any(1.0 + float(np.min(edge_diffs[a][a])) <= 0.05 for a in range(dim)):
    return None
  corner_opts = (slice(None, -1), slice(1, None))
  principal_subsets = list(
      itertools.chain.from_iterable(
          itertools.combinations(range(dim), k) for k in range(2, dim + 1)
      )
  )
  for corner in itertools.product((0, 1), repeat=dim):
    col_slices = [
        tuple(
            slice(None) if a == c else corner_opts[corner[a]]
            for a in range(dim)
        )
        for c in range(dim)
    ]
    for sub in principal_subsets:
      sub_mat = [
          [
              (1.0 if r == c else 0.0) + edge_diffs[r][c][col_slices[c]]
              for c in sub
          ]
          for r in sub
      ]
      if np.min(_det_matrix(sub_mat)) <= 0.05:
        return None

  lin_strides, corner_offsets = _lin_strides_and_offsets(src_spatial)
  u_flats = [u.ravel() for u in u_axes]
  q_axes = [
      np.arange(d, dtype=np.float64).reshape(
          (1,) * a + (d,) + (1,) * (dim - 1 - a)
      )
      for a, d in enumerate(dst_spatial)
  ]
  max_coords = [float(s - 1) for s in src_spatial]

  if math.prod(dst_spatial) >= (16384 << (2 * (dim - 2))):
    c_dev = _invert_newton_jit(
        tuple(jnp.asarray(uf, dtype=jnp.float32) for uf in u_flats),
        jnp.asarray(off_axes, dtype=jnp.float32),
        src_spatial,
        dst_spatial,
    )
    c_axes = [
        np.clip(np.asarray(c_dev[a], dtype=np.float64), 0.0, max_coords[a])
        for a in range(dim)
    ]
    n_iters = 4
  else:
    c_axes = [
        np.clip(
            np.broadcast_to(
                q_axes[a] - off_axes[a] - float(np.mean(u_axes[a])),
                dst_spatial,
            ),
            0.0,
            max_coords[a],
        )
        for a in range(dim)
    ]
    base0 = (
        np.clip(np.rint(c_axes[0]).astype(np.intp), 0, src_spatial[0] - 1)
        * lin_strides[0]
    )
    for a in range(1, dim):
      ka = np.clip(np.rint(c_axes[a]).astype(np.intp), 0, src_spatial[a] - 1)
      base0 = base0 + (ka if lin_strides[a] == 1 else ka * lin_strides[a])
    c_axes = [
        np.clip(q_axes[a] - off_axes[a] - u_flats[a][base0], 0.0, max_coords[a])
        for a in range(dim)
    ]
    n_iters = 9

  for _ in range(n_iters):
    _, nxt_raw = _newton_step_flat(
        np,
        c_axes,
        u_flats,
        off_axes,
        q_axes,
        src_spatial,
        lin_strides,
        corner_offsets,
        np.intp,
        np.float64,
        1e-12,
    )
    nxt = [np.clip(nxt_raw[a], 0.0, max_coords[a]) for a in range(dim)]
    max_delta = max(
        float(np.max(np.abs(nxt[a] - c_axes[a]))) for a in range(dim)
    )
    c_axes = nxt
    if max_delta < 1e-12:
      break

  res_list, nxt_raw = _newton_step_flat(
      np,
      c_axes,
      u_flats,
      off_axes,
      q_axes,
      src_spatial,
      lin_strides,
      corner_offsets,
      np.intp,
      np.float64,
      1e-12,
  )
  res = ft.reduce(np.maximum, (np.abs(r) for r in res_list))
  interior = ft.reduce(
      operator.and_,
      ((c_axes[a] > 0.0) & (c_axes[a] < max_coords[a]) for a in range(dim)),
  )
  inward = ft.reduce(
      operator.and_,
      ((nxt_raw[a] > 0.0) & (nxt_raw[a] < max_coords[a]) for a in range(dim)),
  )
  if np.any((interior | inward) & (res > 1e-8)):
    return None

  valid_out = res <= 1e-8
  out = np.full((dim, *dst_spatial), np.nan, dtype=np.float64)
  for a in range(dim):
    ch = dim - 1 - a
    out[ch] = np.where(
        valid_out,
        ((c_axes[a] + off_axes[a]) - q_axes[a]) * float(stride[a]),
        np.nan,
    )
  return out


def _invert_partial_newton(
    sub_map: np.ndarray,
    off_xyz: Sequence[float],
    stride: Sequence[float],
    dst_spatial: tuple[int, ...],
) -> np.ndarray | None:
  """Inverts a partial-NaN map via EDT-extended Newton plus boundary Qhull."""
  dim = sub_map.shape[0]
  src_spatial = sub_map.shape[1:]
  if any(s < 2 for s in src_spatial) or any(d < 1 for d in dst_spatial):
    return None
  valid_src = np.all(np.isfinite(sub_map), axis=0)
  if np.all(valid_src) or not np.any(valid_src):
    return None
  corner_slices = (slice(None, -1), slice(1, None))
  valid_cell = ft.reduce(
      operator.and_,
      (valid_src[idx] for idx in itertools.product(corner_slices, repeat=dim)),
  )
  if not np.any(valid_cell):
    return None

  edt_idx = ndimage.distance_transform_edt(
      ~valid_src, return_distances=False, return_indices=True
  )
  assert isinstance(edt_idx, np.ndarray)
  edt_sel = tuple(edt_idx)
  filled = np.stack([sub_map[c][edt_sel] for c in range(dim)], axis=0)
  out = _invert_dense_newton(filled, off_xyz, stride, dst_spatial)
  if out is None:
    return None

  q_axes = [
      np.arange(d, dtype=np.float64).reshape(
          (1,) * a + (d,) + (1,) * (dim - 1 - a)
      )
      for a, d in enumerate(dst_spatial)
  ]
  finite_out = np.all(np.isfinite(out), axis=0)
  in_src = finite_out.copy()
  k_axes = []
  for a, s in enumerate(src_spatial):
    ch = dim - 1 - a
    ia = np.where(
        finite_out,
        out[ch] / float(stride[a]) + q_axes[a] - float(off_xyz[ch]),
        -1.0,
    )
    in_src &= (ia >= 0.0) & (ia <= s - 1)
    k_axes.append(
        np.clip(np.floor(np.where(in_src, ia, 0.0)).astype(np.intp), 0, s - 2)
    )

  cell_ok = in_src & valid_cell[tuple(k_axes)]
  out[:, ~cell_ok] = np.nan

  boundary_src = _boundary_valid_mask(valid_src)
  b_idx = np.nonzero(boundary_src)
  b_targets = [
      b_idx[a].astype(np.float64)
      + float(off_xyz[dim - 1 - a])
      + sub_map[(dim - 1 - a, *b_idx)] / float(stride[a])
      for a in range(dim)
  ]
  rem = ~cell_ok
  for q_a, bt_a in zip(q_axes, b_targets):
    rem = rem & (q_a >= float(np.min(bt_a))) & (q_a <= float(np.max(bt_a)))
  if np.any(rem):
    rem_idx = np.nonzero(rem)
    try:
      tri = spatial.Delaunay(
          np.ascontiguousarray(
              np.column_stack(b_targets[::-1]), dtype=np.double
          )
      )
      vals = np.column_stack([-sub_map[(ch, *b_idx)] for ch in range(dim)])
      ip = interpolate.LinearNDInterpolator(tri, vals, fill_value=np.nan)
      query_pts = tuple(idx.astype(np.float64) for idx in rem_idx[::-1])
      out[(slice(None), *rem_idx)] = ip(*query_pts).T
    except spatial.QhullError:
      pass
  return out


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
  if dim not in (2, 3):
    raise NotImplementedError(f'Unsupported coordinate map dim: {dim}')
  stride = _as_vec(stride, dim)
  off_xyz = tuple(
      float(src_box.start[i] - dst_box.start[i]) for i in range(dim)
  )
  src_spatial = rel_map.shape[-dim:]
  dst_spatial = tuple(int(s) for s in dst_box.size[:dim][::-1])
  batch_shape = rel_map.shape[1:-dim]
  n_batch = math.prod(batch_shape)

  work = rel_map.reshape((dim, n_batch, *src_spatial))
  ret = np.full((dim, n_batch, *dst_spatial), np.nan, dtype=np.float64)
  if n_batch == 0 or any(d == 0 for d in dst_spatial):
    return ret.reshape((dim, *batch_shape, *dst_spatial))

  fallback_bs: list[int] = []
  for b in range(n_batch):
    sub = work[:, b]
    if not np.any(np.all(np.isfinite(sub), axis=0)):
      continue
    fast_b = _invert_dense_newton(sub, off_xyz, stride, dst_spatial)
    if fast_b is None:
      fast_b = _invert_partial_newton(sub, off_xyz, stride, dst_spatial)
    if fast_b is not None:
      ret[:, b] = fast_b
    else:
      fallback_bs.append(b)

  if not fallback_bs:
    return ret.reshape((dim, *batch_shape, *dst_spatial))

  # Switch to a coordinate system originating at the first target node
  # of the coordinate map for Delaunay fallback.
  src_box = src_box.adjusted_by(start=-dst_box.start, end=-dst_box.start)  # pyrefly: ignore[bad-argument-type]
  dst_box = dst_box.adjusted_by(start=-dst_box.start, end=-dst_box.start)  # pyrefly: ignore[bad-argument-type]
  abs_map = to_absolute(work[:, fallback_bs], stride, src_box)

  def _box_grid(box: bounding_box.BoundingBox) -> list[np.ndarray]:
    slices = tuple(slice(None, int(s)) for s in box.size[:dim][::-1])
    return [
        (c + start) * step
        for c, start, step in zip(
            np.mgrid[slices], box.start[:dim][::-1], stride
        )
    ]

  src_coords = _box_grid(src_box)  # [z, ]y, x
  query_coords = _box_grid(dst_box)  # [w, ]v, u
  query_points = tuple(q.ravel() for q in query_coords[::-1])  # u, v[, w]

  ret_abs = np.full(
      (dim, len(fallback_bs), *dst_spatial), np.nan, dtype=abs_map.dtype
  )
  for out_i in range(len(fallback_bs)):
    valid = np.all(np.isfinite(abs_map[:, out_i]), axis=0)
    if not np.any(valid):
      continue
    src_points = tuple(c[valid] for c in abs_map[:, out_i])
    try:
      intp = _interpolate_points(
          src_points, query_points, *[s[valid] for s in src_coords[::-1]]
      )
      ret_abs[:, out_i] = intp.reshape((dim, *dst_spatial))
    except spatial.QhullError:
      pass

  ret[:, fallback_bs] = to_relative(ret_abs, stride, dst_box)
  return ret.reshape((dim, *batch_shape, *dst_spatial))


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


def _compose_bilinear_coords(
    xp: Any,
    m1: Any,
    h2: int,
    w2: int,
    ratio: float,
    inv_s2: float,
    off_x: float,
    off_y: float,
    val_dtype: Any,
    idx_dtype: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
  """Computes clamped target cell indices, bilinear weights, and bounds mask."""
  h1, w1 = m1.shape[2], m1.shape[3]
  ix = xp.arange(w1, dtype=val_dtype)[None, None, :]
  iy = xp.arange(h1, dtype=val_dtype)[None, :, None]
  qx = ix * ratio + off_x + m1[0] * inv_s2
  qy = iy * ratio + off_y + m1[1] * inv_s2
  in_b = (
      xp.isfinite(qx)
      & xp.isfinite(qy)
      & (qx >= 0.0)
      & (qx <= float(w2 - 1))
      & (qy >= 0.0)
      & (qy <= float(h2 - 1))
  )
  qx_c = xp.clip(xp.where(in_b, qx, 0.0), 0.0, float(w2 - 1))
  qy_c = xp.clip(xp.where(in_b, qy, 0.0), 0.0, float(h2 - 1))
  kx = xp.clip(xp.floor(qx_c).astype(idx_dtype), 0, w2 - 2)
  ky = xp.clip(xp.floor(qy_c).astype(idx_dtype), 0, h2 - 2)
  wx1 = qx_c - (kx.astype(val_dtype) if xp is jnp else kx)
  wy1 = qy_c - (ky.astype(val_dtype) if xp is jnp else ky)
  return ky, kx, 1.0 - wy1, wy1, 1.0 - wx1, wx1, in_b


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
  ky, kx, wy0, wy1, wx0, wx1, in_b = _compose_bilinear_coords(
      jnp,
      m1,
      m2.shape[2],
      m2.shape[3],
      ratio,
      inv_s2,
      off_x,
      off_y,
      jnp.float32,
      jnp.int32,
  )

  def _sample_z(m1_z, m2_z, ky_z, kx_z, wy0_z, wy1_z, wx0_z, wx1_z, in_b_z):
    v00 = m2_z[:, ky_z, kx_z]
    v01 = m2_z[:, ky_z, kx_z + 1]
    v10 = m2_z[:, ky_z + 1, kx_z]
    v11 = m2_z[:, ky_z + 1, kx_z + 1]
    interp = wy0_z * (wx0_z * v00 + wx1_z * v01) + wy1_z * (
        wx0_z * v10 + wx1_z * v11
    )
    return jnp.where(in_b_z[None, ...], m1_z + interp, jnp.nan)

  return jax.vmap(
      _sample_z, in_axes=(1, 1, 0, 0, 0, 0, 0, 0, 0), out_axes=1
  )(m1, m2, ky, kx, wy0, wy1, wx0, wx1, in_b)


def _compose_maps_bilinear(
    map1: np.ndarray,
    map2: np.ndarray,
    ratio: float,
    inv_s2: float,
    off_x: float,
    off_y: float,
) -> np.ndarray:
  """Evaluates regular-grid bilinear composition across all z slices."""
  nz, h1, w1 = map1.shape[1], map1.shape[2], map1.shape[3]
  m2 = map2[:, :nz, :, :]
  if (
      map1.dtype != np.float64
      and map2.dtype != np.float64
      and nz * h1 * w1 >= 16384
  ):
    out_dev = _compose_maps_dense_jit(
        jnp.asarray(map1, dtype=jnp.float32),
        jnp.asarray(m2, dtype=jnp.float32),
        ratio,
        inv_s2,
        off_x,
        off_y,
    )
    return np.array(out_dev, dtype=map1.dtype, copy=True)

  m1_f = map1.astype(np.float64, copy=False)
  m2_f = m2.astype(np.float64, copy=False)
  h2, w2 = m2.shape[2], m2.shape[3]
  ky, kx, wy0, wy1, wx0, wx1, in_b = _compose_bilinear_coords(
      np,
      m1_f,
      h2,
      w2,
      ratio,
      inv_s2,
      off_x,
      off_y,
      np.float64,
      np.intp,
  )
  base = (np.arange(nz, dtype=np.intp)[:, None, None] * h2 + ky) * w2 + kx
  i00, i01, i10, i11 = base, base + 1, base + w2, base + (w2 + 1)
  composed = np.empty_like(m1_f)
  for ch in range(2):
    flat = np.ascontiguousarray(m2_f[ch]).ravel()
    interp = wy0 * (wx0 * flat[i00] + wx1 * flat[i01]) + wy1 * (
        wx0 * flat[i10] + wx1 * flat[i11]
    )
    composed[ch] = np.where(in_b, m1_f[ch] + interp, np.nan)
  return composed.astype(map1.dtype, copy=False)


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
  if nz == 0 or h1 == 0 or w1 == 0:
    return np.full_like(map1, np.nan)
  assert map2.shape[1] >= nz
  if h2 < 2 or w2 < 2:
    return np.full_like(map1, np.nan)

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

  ret = _compose_maps_bilinear(map1, map2, ratio, inv_s2, off_x, off_y)
  dense_src_z = np.all(np.isfinite(map2[:, :nz, ...]), axis=(0, 2, 3))
  if np.all(dense_src_z):
    return ret

  prev_boundary: np.ndarray | None = None
  prev_tri: spatial.Delaunay | None = None
  for z in np.flatnonzero(~dense_src_z):
    ret_ok = np.all(np.isfinite(ret[:, z]), axis=0)
    if np.all(ret_ok):
      continue
    ret[:, z, ~ret_ok] = np.nan
    rem_cand = np.all(np.isfinite(map1[:, z]), axis=0) & ~ret_ok
    rem_y, rem_x = np.nonzero(rem_cand)
    if rem_y.size == 0:
      continue
    valid_src = np.all(np.isfinite(map2[:, z]), axis=0)
    if not np.any(valid_src):
      continue

    boundary_src = _boundary_valid_mask(valid_src)
    by, bx = np.nonzero(boundary_src)
    m1_rem = map1[:, z, rem_y, rem_x].astype(np.float64)
    qx = rem_x.astype(np.float64) * ratio + off_x + m1_rem[0] * inv_s2
    qy = rem_y.astype(np.float64) * ratio + off_y + m1_rem[1] * inv_s2
    in_b = (
        (qx >= float(bx.min()))
        & (qx <= float(bx.max()))
        & (qy >= float(by.min()))
        & (qy <= float(by.max()))
    )
    if not np.any(in_b):
      continue

    try:
      if (
          prev_boundary is None
          or prev_tri is None
          or not np.array_equal(boundary_src, prev_boundary)
      ):
        prev_tri = spatial.Delaunay(
            np.ascontiguousarray(np.column_stack([bx, by]), dtype=np.double)
        )
        prev_boundary = boundary_src
      vals = map2[:, z, by, bx].astype(np.float64, copy=False).T
      ip = interpolate.LinearNDInterpolator(prev_tri, vals, fill_value=np.nan)
      ret[:, z, rem_y[in_b], rem_x[in_b]] = (
          m1_rem[:, in_b] + ip(qx[in_b], qy[in_b]).T
      ).astype(ret.dtype, copy=False)
    except spatial.QhullError:
      pass

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
