# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Decorators for image warping and rendering."""

from typing import Any, Mapping, MutableMapping, Optional, Sequence

from connectomics.common import bounding_box
from connectomics.common import opencv_utils
from connectomics.volume.decorators import Decorator  # pylint: disable=g-importing-member
import gin
import numpy as np
import scipy.ndimage
import sofima.warp
import tensorstore as ts

JsonSpec = Mapping[str, Any]
MutableJsonSpec = MutableMapping[str, Any]


def _warp_affine(
    img_xyz: np.ndarray,
    matrix_xyz: np.ndarray,
    order: int = 1,
    implementation: str = 'scipy',
    **warp_args):
  """Warp affine in 2D/3D.
  """
  num_img_dims = img_xyz.ndim
  if num_img_dims not in (2, 3):
    raise ValueError(
        f'2 or 3 image dimensions are required, but got {num_img_dims}.')

  num_matrix_dim = matrix_xyz.ndim
  if num_matrix_dim != 2:
    raise ValueError(
        f'2 matrix dimensions are required, but got {num_matrix_dim}.')
  num_matrix_rows, num_matrix_cols = matrix_xyz.shape
  if num_matrix_cols != (num_img_dims + 1):
    raise ValueError(
        f'{num_img_dims + 1} matrix cols are required, ' +
        f'but got {num_matrix_cols}.')
  if num_matrix_rows not in (num_img_dims, num_img_dims + 1):
    raise ValueError(
        f'{num_img_dims} or {num_img_dims + 1} matrix rows are required, ' +
        f'but got {num_matrix_rows}.')
  if num_matrix_rows != num_img_dims + 1:
    matrix_xyzh = np.vstack((
        matrix_xyz, np.array([[0. for _ in range(num_img_dims)] + [1.,]])))
  else:
    matrix_xyzh = matrix_xyz

  if implementation == 'opencv':
    if num_img_dims != 2:
      raise RuntimeError(
          'Only 2D images are supported with `implementation=opencv`.')
    interpolation = None
    for name, value in opencv_utils._INTERPOLATIONS_FLAGS_.items():  # pylint:disable=protected-access
      if value == order:
        interpolation = name
    if not interpolation:
      raise ValueError(
          'Failed finding interpolation corresponding to order value.')
    # Image is transposed since OpenCV uses [x,y]-convention.
    # See `opencv_utils` for more details.
    res_yx = opencv_utils.warp_affine(
        img=img_xyz.T,
        transform=matrix_xyzh[:2, :3],
        interpolation=interpolation)
    return res_yx.T

  elif implementation == 'scipy':
    return scipy.ndimage.affine_transform(
        img_xyz,
        np.linalg.inv(matrix_xyzh),
        order=order)

  elif implementation == 'sofima':
    if num_img_dims != 3:
      raise RuntimeError(
          'Only 3D images are supported with `implementation=sofima`.')
    box = bounding_box.BoundingBox(start=[0, 0, 0], end=img_xyz.shape)
    coord_map = sofima.map_utils.make_affine_map(
        matrix=np.linalg.inv(matrix_xyzh)[:3, :],
        box=box,
        stride=[1, 1, 1])
    if 'work_size' not in warp_args:
      warp_args['work_size'] = img_xyz.shape
    res_zyx = sofima.warp.ndimage_warp(
        image=img_xyz.T,
        coord_map=coord_map,
        stride=[1, 1, 1],
        order=order,
        overlap=[0, 0, 0],
        **warp_args)
    return res_zyx.T

  else:
    raise ValueError(
        'implementation must be `opencv`, `scipy`, or `sofima`, but ' +
        f'got `{implementation}.')


@gin.register
class WarpAffine(Decorator):
  """Warps input TensorStore by 2D/3D affine transformation(s)."""

  def __init__(self,
               transform_spec: JsonSpec,
               image_dims: Sequence[str] = ('x', 'y'),
               context_spec: Optional[MutableJsonSpec] = None,
               **warp_args):
    """Warp affine.

    Note that warping will rechunk the volume.

    Args:
      transform_spec: TensorStore containing affine transformation matrices
        in dimensions labelled `r` (row) and `c` (column). Matrices can be
        2x3 (2D), 3x4 (3D), or homogenous versions thereof (3x3, 4x4).
        Optionally, the TS can have extra dimensions matching ones of the input
        TS in name and size to apply distinct transforms for these dimensions.
      image_dims: Image dimensions to transform, e.g., `x` and `y` (2 or 3).
      context_spec: Spec for virtual chunked context overriding its defaults.
      **warp_args: Passed to `_warp_affine`.
    """
    super().__init__(context_spec)
    self._transform_spec = transform_spec
    self._image_dims = image_dims
    self._warp_args = warp_args

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps the input TensorStore with a virtual_chunked for warping."""

    num_image_dims = len(self._image_dims)
    if num_image_dims not in (2, 3):
      raise ValueError(
          f'2 or 3 image dimensions are required, but got {num_image_dims}.')
    for d in self._image_dims:
      if d not in input_ts.domain.labels:
        raise ValueError(
            f'image dimension {d} not among labels {input_ts.domain.labels}.')
      elif input_ts.domain[d].size < 2:
        raise ValueError(
            'image dimension {d} must at least have size 2 but has size: ' +
            f'{input_ts.domain[d].size}.')

    transform_ts = ts.open(self._transform_spec).result()
    if ('r' or 'c') not in transform_ts.domain.labels:
      raise ValueError(
          'Transform TS needs to contain dimensions `r` and `c` but ' +
          f'contains {transform_ts.domain.labels}.')
    for dim in list(transform_ts.domain):
      if dim.label == 'r':
        if dim.size not in (num_image_dims, num_image_dims + 1):
          raise ValueError(
              f'r-dimension must have size {num_image_dims} or ' +
              f'{num_image_dims + 1} but is {dim.size}.')
      elif dim.label == 'c':
        if dim.size != num_image_dims + 1:
          raise ValueError(
              f'c-dimension must have size {num_image_dims + 1}' +
              f'but is {dim.size}.')
      else:
        if dim.label not in input_ts.domain.labels:
          raise ValueError(
              f'Transform TS contains unmatched dimension: {dim.label}.')
        if dim.size != input_ts.domain[dim.label].size:
          raise ValueError(
              f'Transform TS contains matched {dim.label}-dim of unequal ' +
              f'size: {dim.size} versus {input_ts.domain[dim.label].size}.')

    def warp_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      domain_dict = {dim.label: dim for dim in list(domain)}

      transform_domain = []
      for dim in transform_ts.domain:
        if dim.label in ('r', 'c'):
          transform_domain.append(dim)
        else:
          transform_domain.append(domain_dict[dim.label])
      transform_domain = ts.IndexDomain(transform_domain)

      # Squeeze since we may be batching over an extra dimension.
      array[...] = _warp_affine(
          np.array(input_ts[domain]).squeeze(),
          np.array(transform_ts[transform_domain]).squeeze(),
          **self._warp_args).reshape(array.shape)

    chunksize = []
    for dim in input_ts.domain:
      if dim.label in self._image_dims:
        chunksize.append(dim.size)
      else:
        chunksize.append(1)
    json = input_ts.schema.to_json()
    json['chunk_layout']['read_chunk']['shape'] = chunksize
    json['chunk_layout']['write_chunk']['shape'] = chunksize

    return ts.virtual_chunked(
        warp_fn, schema=ts.Schema(json), context=self._context)


def _warp_coord_map(
    img_xyz: np.ndarray,
    coord_map: np.ndarray,
    **warp_args):
  """Warp by coord map in 3D.
  """
  num_img_dims = img_xyz.ndim
  if num_img_dims != 3:
    raise RuntimeError('Only 3D images are supported.')
  if 'work_size' not in warp_args:
    warp_args['work_size'] = img_xyz.shape
  res_zyx = sofima.warp.ndimage_warp(
      image=img_xyz.T,
      coord_map=coord_map,
      **warp_args)
  return res_zyx.T


@gin.register
class WarpCoordMap(Decorator):
  """Warps input TensorStore with coordinate map."""

  def __init__(self,
               coord_map_spec: JsonSpec,
               image_dims: Sequence[str] = ('x', 'y', 'z'),
               context_spec: Optional[MutableJsonSpec] = None,
               **warp_args):
    """Warp with coordinate map in 3D.

    Note that warping will rechunk the volume.

    Args:
      coord_map_spec: TensorStore containing coordinate map.
      image_dims: Image dimensions to transform, e.g., `x`, `y`, `z` (3).
      context_spec: Spec for virtual chunked context overriding its defaults.
      **warp_args: Passed to `_warp_coord_map`.
    """
    super().__init__(context_spec)
    self._coord_map_spec = coord_map_spec
    self._image_dims = image_dims
    self._warp_args = warp_args

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps the input TensorStore with a virtual_chunked for warping."""

    num_image_dims = len(self._image_dims)
    if num_image_dims != 3:
      raise ValueError(
          f'3 image dimensions are required, but got {num_image_dims}.')
    for d in self._image_dims:
      if d not in input_ts.domain.labels:
        raise ValueError(
            f'image dimension {d} not among labels {input_ts.domain.labels}.')
      elif input_ts.domain[d].size < 2:
        raise ValueError(
            'image dimension {d} must at least have size 2 but has size: ' +
            f'{input_ts.domain[d].size}.')

    coord_map_ts = ts.open(self._coord_map_spec).result()
    for d in ('fc', 'fz', 'fy', 'fx'):
      if d not in coord_map_ts.domain.labels:
        raise ValueError(
            f'coord map dimension {d} not among labels ' +
            f'{coord_map_ts.domain.labels}.')

    def warp_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      domain_dict = {dim.label: dim for dim in list(domain)}

      coord_map_domain = []
      for dim in coord_map_ts.domain:
        if dim.label in ('fc', 'fz', 'fy', 'fx'):
          coord_map_domain.append(dim)
        else:
          coord_map_domain.append(domain_dict[dim.label])
      coord_map_domain = ts.IndexDomain(coord_map_domain)

      # Squeeze since we may be batching over an extra dimension.
      array[...] = _warp_coord_map(
          np.array(input_ts[domain]).squeeze(),
          np.array(coord_map_ts[coord_map_domain]).squeeze(),
          **self._warp_args).reshape(array.shape)

    chunksize = []
    for dim in input_ts.domain:
      if dim.label in self._image_dims:
        chunksize.append(dim.size)
      else:
        chunksize.append(1)
    json = input_ts.schema.to_json()
    json['chunk_layout']['read_chunk']['shape'] = chunksize
    json['chunk_layout']['write_chunk']['shape'] = chunksize

    return ts.virtual_chunked(
        warp_fn, schema=ts.Schema(json), context=self._context)
