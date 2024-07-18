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

import dataclasses
from typing import Optional

from connectomics.common import bounding_box
from connectomics.common import utils
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

  @dataclasses.dataclass(eq=True)
  class EstimateFlowConfig(utils.NPDataClassJsonMixin):
    """Configuration for EstimateFlow."""

    # Patch size in pixels, divisible by 'stride'
    patch_size: int
    # XY stride size in pixels
    stride: int
    # Z stride size in pixels (Δz)
    z_stride: int = 1
    # Whether to compute flow against a fixed current section (first/last
    # section of the subvolume for negative/positive z_stride respectively);
    # this is useful for coming-in regions.
    fixed_current: bool = False
    # mask.MaskConfigs specifying a mask to exclude some voxels from the flow
    # calculation; this mask should have the same resolution and geometry as the
    # input data volume.
    mask_configs: Optional[mask.MaskConfigs] = None
    # Whether to only use mask to decide for which patch pairs to compute flow.
    mask_only_for_patch_selection: bool = False
    # MaskConfigs in text format specifying a mask the positive entries of which
    # indicate locations for which flow should be computed; this mask should
    # have the same resolution and geometry as the output flow volume.
    selection_mask_configs: Optional[mask.MaskConfigs] = None
    # Max number of patches to process in parallel input_volinfo: VolumeInfo for
    # the input volume.
    batch_size: int = 1024

  _config: EstimateFlowConfig

  def __init__(self, config: EstimateFlowConfig, input_volinfo_or_ts_spec=None):
    """Constructor.

    Args:
      config: Parameters for EstimateFlow
      input_volinfo_or_ts_spec: unused
    """

    self._config = config

    assert config.patch_size % config.stride == 0

  def output_type(self, input_type):
    return np.float32

  def subvolume_size(self):
    size = self._config.patch_size * 8
    return subvolume_processor.SuggestedXyz(size, size, 16)

  def context(self):
    config = self._config
    pre = config.patch_size // 2
    post = config.patch_size - pre
    if config.fixed_current:
      if config.z_stride > 0:
        return (pre, pre, 0), (post, post, config.z_stride)
      else:
        return (pre, pre, -config.z_stride), (post, post, 0)
    else:
      if config.z_stride > 0:
        return (pre, pre, config.z_stride), (post, post, 0)
      else:
        return (pre, pre, 0), (post, post, -config.z_stride)

  def num_channels(self, input_channels):
    del input_channels
    return (
        flow_field.JAXMaskedXCorrWithStatsCalculator.non_spatial_flow_channels
        + 2
    )

  def pixelsize(self, psize):
    psize = np.asarray(psize).copy().astype(np.float32)
    psize[:2] *= self._config.stride
    return psize

  def process(self, subvol: subvolume.Subvolume) -> subvolume.Subvolume:
    # TODO(blakely): Determine if Dask supports metrics, and if so, create a
    # shim that supports both Beam and Dask metrics.
    config = self._config

    assert subvol.data.shape[0], 'Input volume should have 1 channel.'
    image = subvol.data[0, ...]
    sel_mask = initial_mask = None

    if config.mask_configs is not None:
      initial_mask = mask.build_mask(config.mask_configs, subvol.bbox)

    if config.selection_mask_configs is not None:
      cropped_bbox = self.crop_box(subvol.bbox)
      sel_start = [
          cropped_bbox.start[0] / config.stride,
          cropped_bbox.start[1] / config.stride,
          subvol.bbox.start[2],
      ]
      xy = np.array([1, 1, 0])
      scale = np.array([config.stride, config.stride, 1])
      sel_size = (
          subvol.bbox.size - xy * config.patch_size + xy * config.stride
      ) / scale
      sel_box = bounding_box.BoundingBox(sel_start, sel_size)
      sel_mask = mask.build_mask(config.selection_mask_configs, sel_box)

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
          config.patch_size,
          config.stride,
          mask_prev,
          mask_curr,
          mask_only_for_patch_selection=config.mask_only_for_patch_selection,
          selection_mask=smask,
          batch_size=config.batch_size,
      )

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flows = []

    if config.fixed_current:
      if config.z_stride > 0:
        rng = range(0, image.shape[0] - 1)
        z_curr = image.shape[0] - 1
      else:
        rng = range(1, image.shape[0])
        z_curr = 0
      for z_prev in rng:
        flows.append(_estimate_flow(z_prev, z_curr))
    else:
      if config.z_stride > 0:
        rng = range(0, image.shape[0] - config.z_stride)
      else:
        rng = range(-config.z_stride, image.shape[0])

      for z in rng:
        flows.append(_estimate_flow(z, z + config.z_stride))

    ret = np.array(flows)

    # Output starts at:
    #   Δz > 0: box.start.z + Δz
    #   Δz < 0: box.start.z
    out_box = self.crop_box(subvol.bbox)
    out_box = bounding_box.BoundingBox(
        start=out_box.start // [config.stride, config.stride, 1],
        size=[ret.shape[-1], ret.shape[-2], out_box.size[2]],
    )

    expected_box = self.expected_output_box(subvol.bbox)
    if out_box != expected_box:
      raise ValueError(
          f'Bounding box does not match expected output_box {out_box} vs '
          f'{expected_box}'
      )

    return subvolume.Subvolume(np.transpose(ret, (1, 0, 2, 3)), out_box)

  # Because mfc.flow_field does not take into account the standard subvolume
  # processor overlap schemes - the latter knows nothing about the internal
  # stride count, so cannot take it into account when calculating overlap - we
  # need to adjust the calculations manually to avoid duplicating output data
  # which can potentially cause write contention. We must modify `overlap` and
  # `expected_output_box` to take this into account.
  def overlap(self) -> subvolume_processor.TupleOrSuggestedXyz:
    overlap = super(EstimateFlow, self).overlap()
    return (
        overlap[0] - self._config.stride,
        overlap[1] - self._config.stride,
        overlap[2],
    )

  def expected_output_box(
      self, box: bounding_box.BoundingBoxBase
  ) -> bounding_box.BoundingBoxBase:

    scale_factor = 1 / self.pixelsize(np.repeat(1, len(box.size)))
    cropped_box = self.crop_box(box)
    scaled_box = cropped_box.scale(list(scale_factor))
    start = scaled_box.start
    size = scaled_box.size
    size[:2] = (
        np.array(self.subvolume_size()[:2])
        - self._config.patch_size
        + self._config.stride
    ) // self._config.stride
    return bounding_box.BoundingBox(start, size)
