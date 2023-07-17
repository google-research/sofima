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
"""Flow field decorators."""

from typing import Any, Mapping, MutableMapping, Optional, Sequence

from connectomics.common import jax_utils
# pylint: disable=g-importing-member
from connectomics.volume.decorators import _adjust_schema_for_chunksize
from connectomics.volume.decorators import Decorator
from connectomics.volume.decorators import Filter
# pylint: enable=g-importing-member
import gin
import jax
import numpy as np
import sofima.flow_field
import sofima.flow_utils
import sofima.mesh
import tensorstore as ts

JsonSpec = Mapping[str, Any]
MutableJsonSpec = MutableMapping[str, Any]


def _clean_flow(flow: np.ndarray, **filter_args) -> np.ndarray:
  """Cleans flow field."""
  final_shape = list(flow.shape)
  final_shape[0] -= 2
  return sofima.flow_utils.clean_flow(
      flow.squeeze(), dim=flow.shape[0] - 2, **filter_args).reshape(final_shape)


@gin.register
class CleanFlowFilter(Filter):
  """Runs filter to clean flow field."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_clean_flow,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)

  def decorate(self, input_ts: ts.TensorStore) -> ts.TensorStore:
    """Wraps input TensorStore with a filtered virtual_chunked."""

    def filt_read(domain: ts.IndexDomain, array: np.ndarray,
                  unused_read_params: ts.VirtualChunkedReadParameters):
      read_domain = list(domain)
      read_domain[0] = ts.Dim(inclusive_min=0,
                              exclusive_max=input_ts.shape[0],
                              label=input_ts.domain.labels[0])
      read_domain = ts.IndexDomain(read_domain)

      array[...] = self._filter_fun(
          np.array(input_ts[read_domain]), **self._filter_args)

    schema = input_ts.schema
    if self._min_chunksize is not None:
      schema = _adjust_schema_for_chunksize(schema, self._min_chunksize)

    # Remove non-spatial dimensions from `fc`-dimension (0th).
    json = schema.to_json()
    json['chunk_layout']['read_chunk']['shape'][0] -= 2
    json['chunk_layout']['write_chunk']['shape'][0] -= 2
    json['domain']['exclusive_max'][0][0] -= 2

    return ts.virtual_chunked(
        filt_read, schema=ts.Schema(json), context=self._context)


def _mesh_relax_flow(flow: np.ndarray, **filter_args) -> np.ndarray:
  """Mesh relaxes flow field."""
  cfg = sofima.mesh.IntegrationConfig(**filter_args)
  x = np.zeros_like(flow.squeeze())

  num_spatial_dim = flow.shape[0]
  if num_spatial_dim == 2:
    res = sofima.mesh.relax_mesh(x, flow.squeeze(), cfg)
  elif num_spatial_dim == 3:
    res = sofima.mesh.relax_mesh(x, flow.squeeze(), cfg,
                                 mesh_force=sofima.mesh.elastic_mesh_3d)
  else:
    raise ValueError(
        f'`num_spatial_dim` must be 2 or 3 but is {num_spatial_dim}.')

  return np.asarray(res[0]).reshape(flow.shape)


@gin.register
class MeshRelaxFlowFilter(Filter):
  """Runs filter to mesh relax flow field."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_mesh_relax_flow,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)


def _flow_shape(o, p, s):
  return np.ceil((o - p + 1) / s).astype(int)


def _padded_flow_shape(o, p, s):
  return _flow_shape(o, p, s) + p // s - 1


@gin.register
class OptimFlow(Decorator):
  """Finds 2D/3D flow for registration via cross-correlation."""

  def __init__(self,
               fixed_spec: JsonSpec,
               image_dims: Sequence[str] = ('x', 'y'),
               context_spec: Optional[MutableJsonSpec] = None,
               patch_size: Sequence[int] = (32, 32),
               step_size: Sequence[int] = (16, 16),
               batch_size: int = 1,
               pad: bool = True,
               input_mask_spec: Optional[JsonSpec] = None,
               fixed_mask_spec: Optional[JsonSpec] = None,
               invert_masks: bool = False,
               jax_device: Optional[str] = None,
               **flow_args):
    """Computes flow between volumes for registration.

    Uses sofima to compute flow fields between two volumes, where 2D or 3D
    images are taken from the input TensorStore and fixed images from the
    one specified by `fixed_spec`.

    The resulting TensorStore has at least the following dimensions:
      `fc`: Indexes into flow field components Δx, Δy, (Δz for 3D), and two
        channels containing peak information.
      `fz`: If three image dimensions are specified, this corresponds to the
        third one. Otherwise this dimension is of size 1.
      `fy`: Corresponds to the second image dimension.
      `fx`: Corresponds to the first image dimension.

    If the input TensorStore has dimensions that are not specified as part of
    `image_dims`, they become trailing dimensions of the resulting TensorStore.

    Args:
      fixed_spec: TensorStore containing fixed images to align against.
        Must have same dimensions as input TS (labels and shape).
      image_dims: Image dimensions to transform, e.g., `x` and `y` (2 or 3).
      context_spec: Spec for virtual chunked context overriding its defaults.
      patch_size: Patch size for flow field computation, in the same order
        that image dimensions are specified through `image_dims`.
      step_size: Step size for flow field computation, in the same order
        that image dimensions are specified through `image_dims`.
      batch_size: Batch size for flow field computation.
      pad: If True, will pad outputs for further processing steps.
      input_mask_spec: TensorStore containing mask for input images.
        Must have same dimensions as input TS (labels and shape).
      fixed_mask_spec: TensorStore containing mask for fixed images.
        Must have same dimensions as fixed TS (labels and shape).
      invert_masks: By default, masks identify regions in input and fixed
        volumes that should not be considered for flow calculation (1s).
        If this flag is set, masks are inverted before flow calculation.
      jax_device: If set, used as device for computation.
      **flow_args: Passed to `flow_field` estimation method.
    """
    super().__init__(context_spec)
    self._fixed_spec = fixed_spec
    self._image_dims = image_dims
    self._patch_zyx = patch_size[::-1]  # [z,]yx
    self._step_zyx = step_size[::-1]  # [z,]yx
    self._batch_size = batch_size
    self._pad = pad
    self._input_mask_spec = input_mask_spec
    self._fixed_mask_spec = fixed_mask_spec
    self._invert_masks = invert_masks
    self._jax_device = jax_utils.parse_device_str(
        jax_device) if jax_device is not None else None
    self._flow_args = flow_args

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

    if self._input_mask_spec is not None:
      input_mask_ts = ts.open(self._input_mask_spec).result()
      if input_ts.domain.labels != input_mask_ts.domain.labels:
        raise ValueError(
            'Input TS and input mask TS must have same labels, but they are ' +
            f'{input_ts.domain.labels} and {input_mask_ts.domain.labels}.')
      if input_ts.shape != input_mask_ts.shape:
        raise ValueError(
            'Input TS and input mask TS must have same shape, but they are ' +
            f'{input_ts.shape} and {input_mask_ts.shape}.')

    if self._fixed_mask_spec is not None:
      fixed_mask_ts = ts.open(self._fixed_mask_spec).result()
      if input_ts.domain.labels != fixed_mask_ts.domain.labels:
        raise ValueError(
            'Input TS and input mask TS must have same labels, but they are ' +
            f'{input_ts.domain.labels} and {fixed_mask_ts.domain.labels}.')
      if input_ts.shape != fixed_mask_ts.shape:
        raise ValueError(
            'Input TS and input mask TS must have same shape, but they are ' +
            f'{input_ts.shape} and {fixed_mask_ts.shape}.')

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

      pre_mask = None
      if self._input_mask_spec is not None:
        pre_mask = np.array(
            input_mask_ts[read_domain], dtype=bool).squeeze().T
        if self._invert_masks:
          pre_mask = ~pre_mask  # pylint: disable=invalid-unary-operand-type

      post_mask = None
      if self._fixed_mask_spec is not None:
        post_mask = np.array(
            fixed_mask_ts[read_domain], dtype=bool).squeeze().T
        if self._invert_masks:
          post_mask = ~post_mask  # pylint: disable=invalid-unary-operand-type

      with jax.default_device(self._jax_device):
        mfc = sofima.flow_field.JAXMaskedXCorrWithStatsCalculator()
        flow_post_to_pre = mfc.flow_field(
            pre_image=np.array(input_ts[read_domain],
                               dtype=np.float32).squeeze().T,
            post_image=np.array(fixed_ts[read_domain],
                                dtype=np.float32).squeeze().T,
            pre_mask=pre_mask,
            post_mask=post_mask,
            patch_size=self._patch_zyx,
            step=self._step_zyx,
            batch_size=self._batch_size,
            **self._flow_args)

      if num_image_dims == 2:
        flow_post_to_pre = np.asarray(flow_post_to_pre[:, np.newaxis, ...])

      if self._pad:
        pad_total = np.array(self._patch_zyx) // np.array(self._step_zyx) - 1
        pad_left = np.array(self._patch_zyx) // np.array(self._step_zyx) // 2
        pad_width = [(0, 0)]
        if num_image_dims == 2:
          pad_width.append([0, 0])
        for left, total in zip(pad_left, pad_total):
          pad_width.append([left, total - left])
        array[...] = np.pad(
            flow_post_to_pre, pad_width, constant_values=np.nan
        ).reshape(array.shape)
      else:
        array[...] = flow_post_to_pre.reshape(array.shape)

    labels = ['fc', 'fz', 'fy', 'fx'] + non_image_dims

    flow_shape = {}
    flow_shape['fc'] = num_image_dims + 2
    if num_image_dims == 2:
      flow_shape['fz'] = 1
    calc_shape = _padded_flow_shape if self._pad else _flow_shape
    for i, l in enumerate(self._image_dims):
      flow_shape[labels[3 - i]] = calc_shape(
          o=input_domain_dict[l].size,
          p=self._patch_zyx[-1 - i],
          s=self._step_zyx[-1 - i])

    chunksize = []
    for l in labels:
      if l in non_image_dims:
        chunksize.append(1)
      else:
        chunksize.append(flow_shape[l])

    schema = {
        'chunk_layout': {
            'read_chunk': {'shape': chunksize},
            'write_chunk': {'shape': chunksize},
        },
        'domain': {
            'labels': labels,
            'inclusive_min': [0 for _ in labels[:4]] + [
                input_domain_dict[l].inclusive_min for l in non_image_dims],
            'exclusive_max': [flow_shape[l] for l in labels[:4]] + [
                input_domain_dict[l].exclusive_max for l in non_image_dims],
        },
        'dtype': 'float32',
        'rank': len(chunksize),
    }

    return ts.virtual_chunked(
        read_fn, schema=ts.Schema(schema), context=self._context)


def _reconcile_flow(
    flow: np.ndarray, **filter_args
) -> np.ndarray:
  """Reconciles single flow field."""
  return sofima.flow_utils.reconcile_flows(
      [flow.squeeze()], **filter_args).reshape(flow.shape)


@gin.register
class ReconcileFlowFilter(Filter):
  """Runs filter to reconcile a single flow field."""

  def __init__(self,
               min_chunksize: Optional[Sequence[int]] = None,
               context_spec: Optional[MutableJsonSpec] = None,
               **filter_args):
    super().__init__(
        filter_fun=_reconcile_flow,
        context_spec=context_spec,
        min_chunksize=min_chunksize,
        **filter_args)
