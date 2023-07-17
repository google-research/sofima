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
"""Affine transform decorators."""

from typing import Any, Mapping, MutableMapping, Optional, Sequence

from connectomics.common import opencv_utils
from connectomics.volume.decorators import Decorator  # pylint: disable=g-importing-member
import gin
import numpy as np
import skimage.feature
import tensorstore as ts

JsonSpec = Mapping[str, Any]
MutableJsonSpec = MutableMapping[str, Any]


@gin.register
class OptimAffineTransformSectionwise(Decorator):
  """Finds 2D affine transforms sectionwise by ECC optimization."""

  def __init__(self,
               fixed_spec: JsonSpec,
               image_dims: Sequence[str] = ('x', 'y'),
               batch_dim: Optional[str] = None,
               init_previous: bool = False,
               context_spec: Optional[MutableJsonSpec] = None,
               **optim_args):
    """Optimize affine transform sectionwise.

    Uses OpenCV's `cv.findTransformECC` to find an affine transformations that
    aligns moving and fixed 2D images, where moving images are taken from the
    input TensorStore and fixed images from the one specified by `fixed_spec`.

    Note that optimisation is done per 2D section according to `image_dims`.
    The resulting TensorStore contains 2x3 transformation matrices in
    dimensions 'r' (row) and 'c' (column), for all non-image dimensions.
    Transformation matrices are stored in individual chunks.

    Args:
      fixed_spec: TensorStore containing fixed images to align against.
        Must have same dimensions as input TS (labels and shape).
      image_dims: Image dimensions to transform, e.g., `x` and `y` (two).
      batch_dim: Optional dimension to batch reads.
      init_previous: If True, initializes transform for subsequent calls
        of the optimization function with the previous result. Requires
        specification of a `batch_dim`.
      context_spec: Spec for virtual chunked context overriding its defaults.
      **optim_args: Passed to `opencv_utils.optim_transform`.
    """
    super().__init__(context_spec)
    self._fixed_spec = fixed_spec
    self._image_dims = image_dims
    self._batch_dim = batch_dim
    self._init_previous = init_previous
    if init_previous and not batch_dim:
      raise ValueError('`batch_dim` must be specified to use `init_previous`.')
    if 'transform_initial' in optim_args:
      self._transform_initial = optim_args['transform_initial']
      optim_args.pop('transform_initial')
    else:
      self._transform_initial = None
    self._optim_args = optim_args

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps input TensorStore with a virtual_chunked for optim_transform."""

    fixed_ts = ts.open(self._fixed_spec).result()
    if input_ts.domain.labels != fixed_ts.domain.labels:
      raise ValueError(
          'Input TS and fixed TS must have same labels, but they are ' +
          f'{input_ts.domain.labels} and {fixed_ts.domain.labels}.')
    if input_ts.shape != fixed_ts.shape:
      raise ValueError(
          'Input TS and fixed TS must have same shape, but they are ' +
          f'{input_ts.shape} and {fixed_ts.shape}.')

    if len(self._image_dims) != 2:
      raise ValueError(
          f'2 image dimensions are required, but got {len(self._image_dims)}.')
    for d in self._image_dims:
      if d not in input_ts.domain.labels:
        raise ValueError(
            f'image dimension {d} not among labels {input_ts.domain.labels}.')
      elif input_ts.domain[d].size < 2:
        raise ValueError(
            'image dimension {d} must at least have size 2 but has size: ' +
            f'{input_ts.domain[d].size}.')

    non_image_dims = [
        l for l in input_ts.domain.labels if l not in self._image_dims]
    input_domain_dict = {dim.label: dim for dim in list(input_ts.domain)}
    batch_idx = (input_ts.domain.labels.index(self._batch_dim)
                 if self._batch_dim else None)

    def read_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      domain_dict = {dim.label: dim for dim in list(domain)}

      if self._transform_initial:
        transform_initial = self._transform_initial.copy()
      else:
        transform_initial = None

      if not self._batch_dim:
        read_domain = []
        for l in non_image_dims:
          read_domain.append(domain_dict[l])
        for l in self._image_dims:
          read_domain.append(input_domain_dict[l])
        read_domain = ts.IndexDomain(read_domain)

        # Images are transposed since OpenCV uses [x,y]-convention.
        # See `opencv_utils` for more details.
        _, transform = opencv_utils.optim_transform(
            fix=np.array(fixed_ts[read_domain], dtype=np.float32).squeeze().T,
            mov=np.array(input_ts[read_domain], dtype=np.float32).squeeze().T,
            transform_initial=transform_initial,
            **self._optim_args)

        array[...] = transform.reshape(array.shape)
      else:
        for i, j in enumerate(domain_dict[self._batch_dim]):
          read_domain = []
          for l in non_image_dims:
            if l != self._batch_dim:
              read_domain.append(domain_dict[l])
            else:
              read_domain.append(
                  ts.Dim(inclusive_min=j, exclusive_max=j+1,
                         label=self._batch_dim))
          for l in self._image_dims:
            read_domain.append(input_domain_dict[l])
          read_domain = ts.IndexDomain(read_domain)

          # Images are transposed since OpenCV uses [x,y]-convention.
          # See `opencv_utils` for more details.
          _, transform = opencv_utils.optim_transform(
              fix=np.array(fixed_ts[read_domain], dtype=np.float32).squeeze().T,
              mov=np.array(input_ts[read_domain], dtype=np.float32).squeeze().T,
              transform_initial=transform_initial,
              **self._optim_args)
          if self._init_previous:
            transform_initial = transform

          idx = [slice(None) for _ in range(array.ndim)]
          idx[batch_idx] = i
          array[tuple(idx)] = transform.reshape(array[tuple(idx)].shape)

    chunksize = [2, 3]
    for l in non_image_dims:
      if l != self._batch_dim:
        chunksize.append(1)
      else:
        chunksize.append(input_domain_dict[l].size)
    schema = {
        'chunk_layout': {
            'read_chunk': {'shape': chunksize},
            'write_chunk': {'shape': chunksize},
        },
        'domain': {
            'labels': ['r', 'c',] + non_image_dims,
            'inclusive_min': [0, 0] + [
                input_domain_dict[l].inclusive_min for l in non_image_dims],
            'exclusive_max': [2, 3] + [
                input_domain_dict[l].exclusive_max for l in non_image_dims],
        },
        'dtype': 'float64',
        'rank': len(chunksize),
    }

    return ts.virtual_chunked(
        read_fn, schema=ts.Schema(schema), context=self._context)


@gin.register
class OptimTranslationTransform(Decorator):
  """Finds 2D/3D translations for registration via cross-correlation."""

  def __init__(self,
               fixed_spec: JsonSpec,
               image_dims: Sequence[str] = ('x', 'y'),
               context_spec: Optional[MutableJsonSpec] = None,
               **optim_args):
    """Computes cross-correlation between volumes for registration.

    Uses skimage's `registration.phase_cross_correlation` to find translation
    matrices for registration of two volumes, where 2D or 3D moving images are
    taken from the input TensorStore and fixed images from the one specified by
    `fixed_spec`.

    The resulting TensorStore contains 2x3 (2D) or 3x4 (3D) transformation
    matrices in dimensions 'r' (row) and 'c' (column), for all non-image
    dimensions. Transformation matrices are stored in individual chunks.

    Args:
      fixed_spec: TensorStore containing fixed images to align against.
        Must have same dimensions as input TS (labels and shape).
      image_dims: Image dimensions to transform, e.g., `x` and `y` (two).
      context_spec: Spec for virtual chunked context overriding its defaults.
      **optim_args: Passed to `skimage.registration.phase_cross_correlation`.
    """
    super().__init__(context_spec)
    self._fixed_spec = fixed_spec
    self._image_dims = image_dims
    self._optim_args = optim_args

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps input TensorStore with a virtual_chunked."""

    fixed_ts = ts.open(self._fixed_spec).result()
    if input_ts.domain.labels != fixed_ts.domain.labels:
      raise ValueError(
          'Input TS and fixed TS must have same labels, but they are ' +
          f'{input_ts.domain.labels} and {fixed_ts.domain.labels}.')
    if input_ts.shape != fixed_ts.shape:
      raise ValueError(
          'Input TS and fixed TS must have same shape, but they are ' +
          f'{input_ts.shape} and {fixed_ts.shape}.')

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

    non_image_dims = [
        l for l in input_ts.domain.labels if l not in self._image_dims]
    input_domain_dict = {dim.label: dim for dim in list(input_ts.domain)}

    def read_fn(domain: ts.IndexDomain, array: np.ndarray,
                unused_read_params: ts.VirtualChunkedReadParameters):
      domain_dict = {dim.label: dim for dim in list(domain)}

      read_domain = []
      for l in non_image_dims:
        read_domain.append(domain_dict[l])
      for l in self._image_dims:
        read_domain.append(input_domain_dict[l])
      read_domain = ts.IndexDomain(read_domain)

      # Default to no normalization.
      if 'normalization' not in self._optim_args:
        self._optim_args['normalization'] = None

      translation, _, _ = skimage.registration.phase_cross_correlation(
          reference_image=np.array(
              fixed_ts[read_domain], dtype=np.float32).squeeze(),
          moving_image=np.array(
              input_ts[read_domain], dtype=np.float32).squeeze(),
          **self._optim_args)
      transform = np.hstack(
          (np.eye(len(self._image_dims)), translation.reshape(-1, 1)))

      array[...] = transform.reshape(array.shape)

    chunksize = [num_image_dims, num_image_dims + 1]
    for _ in non_image_dims:
      chunksize.append(1)
    schema = {
        'chunk_layout': {
            'read_chunk': {'shape': chunksize},
            'write_chunk': {'shape': chunksize},
        },
        'domain': {
            'labels': ['r', 'c',] + non_image_dims,
            'inclusive_min': [0, 0] + [
                input_domain_dict[l].inclusive_min for l in non_image_dims],
            'exclusive_max': [num_image_dims, num_image_dims + 1] + [
                input_domain_dict[l].exclusive_max for l in non_image_dims],
        },
        'dtype': 'float64',
        'rank': len(chunksize),
    }

    return ts.virtual_chunked(
        read_fn, schema=ts.Schema(schema), context=self._context)
