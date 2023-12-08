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
"""Decorators for coordinate maps."""

from typing import Any, Mapping, MutableMapping, Optional, Sequence

from connectomics.common import bounding_box
# pylint: disable=g-importing-member
from connectomics.volume.decorators import adjust_schema_for_virtual_chunked
from connectomics.volume.decorators import Decorator
# pylint: enable=g-importing-member
import gin
import numpy as np
import sofima
import sofima.map_utils
import tensorstore as ts

JsonSpec = Mapping[str, Any]
MutableJsonSpec = MutableMapping[str, Any]


@gin.register
class ComposeCoordMaps(Decorator):
  """Compose coordinate maps stored in TensorStores."""

  def __init__(self,
               coord_map_spec: JsonSpec,
               context_spec: Optional[MutableJsonSpec] = None,
               **compose_args):
    """Composes coordinate maps.

    Uses sofima's `map_utils.compose_maps_fast` to compose coordinate maps.

    Note that this will rechunk the volume.

    Args:
      coord_map_spec: Spec of coordinate map to compose input with.
      context_spec: Spec for virtual chunked context overriding its defaults.
      **compose_args: Passed to `map_utils.compose_maps_fast`.
    """
    super().__init__(context_spec)
    self._coord_map_spec = coord_map_spec
    self._compose_args = compose_args

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps the input TensorStore with a virtual_chunked."""

    coord_map_ts = ts.open(self._coord_map_spec).result()
    compose_maps = sofima.map_utils.compose_maps_fast

    for d in ('fc', 'fz', 'fy', 'fx'):
      if d not in coord_map_ts.domain.labels:
        raise ValueError(
            f'coord map dimension {d} not among labels ' +
            f'{coord_map_ts.domain.labels}.')
    if input_ts.domain.labels != coord_map_ts.domain.labels:
      raise ValueError(
          'Input TS and coord map TS must have same labels, but they are ' +
          f'{input_ts.domain.labels} and {coord_map_ts.domain.labels}.')
    if input_ts.shape != coord_map_ts.shape:
      raise ValueError(
          'Input TS and coord map TS must have same shape, but they are ' +
          f'{input_ts.shape} and {coord_map_ts.shape}.')

    def warp_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      array[...] = compose_maps(
          map1=np.array(input_ts[domain]).squeeze(),
          map2=np.array(coord_map_ts[domain]).squeeze(),
          **self._compose_args).reshape(array.shape)

    chunksize = []
    for dim in input_ts.domain:
      if dim.label in ('fc', 'fz', 'fy', 'fx'):
        chunksize.append(dim.size)
      else:
        chunksize.append(1)
    schema = adjust_schema_for_virtual_chunked(input_ts.schema)
    json = schema.to_json()
    json['chunk_layout']['read_chunk']['shape'] = chunksize
    json['chunk_layout']['write_chunk']['shape'] = chunksize

    return ts.virtual_chunked(
        warp_fn, schema=ts.Schema(json), context=self._context)


@gin.register
class MakeAffineCoordMap(Decorator):
  """Makes affine coordinate map."""

  def __init__(self,
               size: Sequence[int],
               context_spec: Optional[MutableJsonSpec] = None,):
    """Make affine coordinate map from 3D affine transforms.

    Uses sofima's `make_affine_map` to create a dense coordinate map.

    The input TensorStore is assumed to contain 3x4 (3D) transformation matrices
    in dimensions 'r' (row) and 'c' (column).

    The resulting TensorStore has at least the following dimensions:
      `fc`: Indexes into flow field components Δx, Δy, Δz.
      `fz`: Corresponds to the third image dimension.
      `fy`: Corresponds to the second image dimension.
      `fx`: Corresponds to the first image dimension.

    If the input TensorStore has dimensions beyond 'r' and 'c', they become
    trailing dimensions of the resulting TensorStore.

    Note that this will rechunk the volume.

    Args:
      size: Size of the coordinate map in x, y, and z.
      context_spec: Spec for virtual chunked context overriding its defaults.
    """
    super().__init__(context_spec)
    self._size_xyz = size
    self._start_xyz = (0, 0, 0)
    self._stride_zyx = (1, 1, 1)
    self._transform_dims = ('r', 'c')

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps input TensorStore with a virtual_chunked."""

    for d in self._transform_dims:
      if d not in input_ts.domain.labels:
        raise ValueError(
            f'transform dimension {d} not among labels '
            + f'{input_ts.domain.labels}.')

    non_transform_dims = [
        l for l in input_ts.domain.labels if l not in self._transform_dims]
    input_domain_dict = {dim.label: dim for dim in list(input_ts.domain)}

    box = bounding_box.BoundingBox(start=self._start_xyz, size=self._size_xyz)

    def read_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      domain_dict = {dim.label: dim for dim in list(domain)}

      read_domain = []
      for d in self._transform_dims:
        read_domain.append(input_domain_dict[d])
      for d in non_transform_dims:
        read_domain.append(domain_dict[d])
      read_domain = ts.IndexDomain(read_domain)

      matrix_xyz = np.array(input_ts[read_domain], dtype=np.float32).squeeze()
      coord_map = sofima.map_utils.make_affine_map(
          matrix_xyz, box, self._stride_zyx)
      array[...] = coord_map.reshape(array.shape)

    chunksize = [3,] + list(box.size)[::-1]
    for l in non_transform_dims:
      chunksize.append(1)
    schema = {
        'chunk_layout': {
            'read_chunk': {'shape': chunksize},
            'write_chunk': {'shape': chunksize},
        },
        'domain': {
            'labels': ['fc', 'fz', 'fy', 'fx'] + non_transform_dims,
            'inclusive_min': [0, 0, 0, 0] + [
                input_domain_dict[l].inclusive_min for l in non_transform_dims],
            'exclusive_max': chunksize[:4] + [
                input_domain_dict[l].exclusive_max for l in non_transform_dims],
        },
        'dtype': 'float32',
        'rank': len(chunksize),
    }

    return ts.virtual_chunked(
        read_fn, schema=ts.Schema(schema), context=self._context)
