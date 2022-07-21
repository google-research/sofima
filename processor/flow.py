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
"""Flow field estimation from SOFIMA."""

from typing import Optional

from connectomics.common import bounding_box
from connectomics.volume import mask
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import numpy as np
from sofima import flow_field


class EstimateFlow(subvolume_processor.SubvolumeProcessor):
  """Estimates section-to-section optical flow.

  The flow f(z) for the 'current section' at 'z' always defines how the points
  at 'z' should be moved in order to match points in the 'reference section'
  at 'z - Δz', i.e.:

       p(z) + f(z) <-> p(z - Δz)

       z - Δz   -----.---   reference
                     ^
                    / f(z)  flow vector
                  /
            z   -.-------   current

  Δz > 0 indicates an earlier reference section (forward flow), whereas
  Δz < 0 indicates a later reference section (backward flow).

  The output data is organized so that the flow value estimated for the patch
  centered at 'x' is stored at 'x' // patch_size.
  """

  _patch_size: int
  _stride: int
  _z_stride: int
  _fixed_current: bool
  _batch_size: int
  _mask_config: Optional[mask.MaskConfigs]
  _sel_mask_config: Optional[mask.MaskConfigs]
  _mask_only_for_patch_selection: bool

  def __init__(self,
               patch_size: int,
               stride: int,
               z_stride: int = 1,
               fixed_current: bool = False,
               mask_configs: Optional[mask.MaskConfigs] = None,
               mask_only_for_patch_selection: bool = False,
               selection_mask_configs=None,
               batch_size=1024,
               input_volinfo=None):
    """Constructor.

    Args:
      patch_size: patch size in pixels, divisible by 'stride'
      stride: XY stride size in pixels
      z_stride: Z stride size in pixels (Δz)
      fixed_current: whether to compute flow against a fixed current section
        (first/last section of the subvolume for negative/positive z_stride
        respectively); this is useful for coming-in regions
      mask_configs: MaskConfigs proto in text format specifying a mask to
        exclude some voxels from the flow calculation
      mask_only_for_patch_selection: whether to only use mask to decide for
        which patch pairs to compute flow
      selection_mask_configs: MaskConfigs in text format specifying a mask the
        positive entries of which indicate locations for which flow should be
        computed; this mask should have the same resolution and geometry as the
        output flow volume
      batch_size: max number of patches to process in parallel
      input_volinfo: VolumeInfo for the input volume
    """
    self._patch_size = patch_size
    self._stride = stride
    assert self._patch_size % self._stride == 0

    self._z_stride = z_stride
    self._fixed_current = fixed_current
    self._batch_size = batch_size

    self._mask_config = mask_configs

    self._sel_mask_config = selection_mask_configs

    self._mask_only_for_patch_selection = mask_only_for_patch_selection
    self.input_volinfo = input_volinfo

  def output_type(self, input_type):
    return np.float32

  def subvolume_size(self):
    size = self._patch_size * 8
    return subvolume_processor.SuggestedXyz(size, size, 16)

  def context(self):
    pre = self._patch_size // 2
    post = self._patch_size - pre
    if self._fixed_current:
      if self._z_stride > 0:
        return (pre, pre, 0), (post, post, self._z_stride)
      else:
        return (pre, pre, -self._z_stride), (post, post, 0)
    else:
      if self._z_stride > 0:
        return (pre, pre, self._z_stride), (post, post, 0)
      else:
        return (pre, pre, 0), (post, post, -self._z_stride)

  def num_channels(self, input_channels):
    del input_channels
    return flow_field.JAXMaskedXCorrWithStatsCalculator.non_spatial_flow_channels + 2

  def pixelsize(self, psize):
    psize = np.asarray(psize).copy().astype(np.float32)
    psize[:2] *= self._stride
    return psize

  def process(self, subvol: subvolume.Subvolume) -> subvolume.Subvolume:
    # TODO(blakely): Determine if Dask supports metrics, and if so, create a
    # shim that supports both Beam and Dask metrics.

    assert subvol.data.shape[0], 'Input volume should have 1 channel.'
    image = subvol.data[0, ...]
    sel_mask = initial_mask = None

    if self._mask_config is not None:
      initial_mask = mask.build_mask(self._mask_config, subvol.bbox)

      if self._sel_mask_config is not None:
        sel_box = subvol.bbox.scale([1.0 / self._stride, 1.0 / self._stride, 1])
        sel_mask = mask.build_mask(self._sel_mask_config, sel_box)

    def _estimate_flow(z_prev, z_curr):
      mask_prev = mask_curr = None
      prev = image[z_prev, ...]
      curr = image[z_curr, ...]

      if initial_mask is not None:
        mask_prev = initial_mask[z_prev, ...]
        mask_curr = initial_mask[z_curr, ...]

      smask = None
      if sel_mask is not None:
        smask = sel_mask[z_curr, ...]

      return mfc.flow_field(
          prev,
          curr,
          self._patch_size,
          self._stride,
          mask_prev,
          mask_curr,
          mask_only_for_patch_selection=self._mask_only_for_patch_selection,
          selection_mask=smask,
          batch_size=self._batch_size)

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flows = []

    if self._fixed_current:
      if self._z_stride > 0:
        rng = range(0, image.shape[0] - 1)
        z_curr = image.shape[0] - 1
      else:
        rng = range(1, image.shape[0])
        z_curr = 0
      for z_prev in rng:
        flows.append(_estimate_flow(z_prev, z_curr))
    else:
      if self._z_stride > 0:
        rng = range(0, image.shape[0] - self._z_stride)
      else:
        rng = range(-self._z_stride, image.shape[0])

      for z in rng:
        flows.append(_estimate_flow(z, z + self._z_stride))

    ret = np.array(flows)

    # Output starts at:
    #   Δz > 0: box.start.z + Δz
    #   Δz < 0: box.start.z
    out_box = self.crop_box(subvol.bbox)
    out_box = bounding_box.BoundingBox(
        start=out_box.start // [self._stride, self._stride, 1],
        size=[ret.shape[-1], ret.shape[-2], out_box.size[2]])

    expected_box = self.expected_output_box(subvol.bbox)
    if out_box != expected_box:
      raise ValueError(
          f'Bounding box does not match expected output_box {out_box} vs '
          f'{expected_box}')

    return subvolume.Subvolume(np.transpose(ret, (1, 0, 2, 3)), out_box)

  def expected_output_box(
      self, box: bounding_box.BoundingBoxBase) -> bounding_box.BoundingBoxBase:
    scale_factor = 1 / self.pixelsize(np.array([1, 1, 1]))
    cropped_box = self.crop_box(box)
    return cropped_box.scale(list(scale_factor))
