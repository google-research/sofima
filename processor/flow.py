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
import time
from typing import Any, Sequence

from absl import logging
from connectomics.common import beam_utils
from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.common import utils
from connectomics.volume import base
from connectomics.volume import mask as mask_lib
from connectomics.volume import metadata
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import dataclasses_json
import numpy as np
from scipy import interpolate
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils


Subvolume = subvolume.Subvolume
SubvolumeOrMany = Subvolume | list[Subvolume]


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

  @dataclasses_json.dataclass_json
  @dataclasses.dataclass(eq=True)
  class EstimateFlowConfig(utils.NPDataClassJsonMixin):
    """Configuration for EstimateFlow.

    Attributes:
      patch_size: Patch size in pixels, divisible by 'stride'
      stride: XY stride size in pixels
      z_stride: Z stride size in pixels (Δz)
      fixed_current: Whether to compute flow against a fixed current section
        (first/last section of the subvolume for negative/positive z_stride
        respectively); this is useful for coming-in regions.
      mask_configs: mask.MaskConfigs specifying a mask to exclude some voxels
        from the flow calculation; this mask should have the same resolution and
        geometry as the input data volume.
      mask_only_for_patch_selection: Whether to only use mask to decide for
        which patch pairs to compute flow.
      selection_mask_configs: MaskConfigs in text format specifying a mask the
        positive entries of which indicate locations for which flow should be
        computed; this mask should have the same resolution and geometry as the
        output flow volume.
      batch_size: Max number of patches to process in parallel.
    """

    patch_size: int = 160
    stride: int = 40
    z_stride: int = 1
    fixed_current: bool = False
    mask_configs: str | mask_lib.MaskConfigs | None = None
    mask_only_for_patch_selection: bool = False
    selection_mask_configs: mask_lib.MaskConfigs | None = None
    batch_size: int = 1024

  _config: EstimateFlowConfig

  def __init__(self, config: EstimateFlowConfig, input_volinfo_or_ts_spec=None):
    """Constructor.

    Args:
      config: Parameters for EstimateFlow
      input_volinfo_or_ts_spec: unused
    """

    del input_volinfo_or_ts_spec
    self._config = config
    assert config.patch_size % config.stride == 0

    if config.mask_configs is not None:
      if isinstance(config.mask_configs, str):
        config.mask_configs = self._get_mask_configs(config.mask_configs)

    if config.selection_mask_configs is not None:
      if isinstance(config.selection_mask_configs, str):
        config.selection_mask_configs = self._get_mask_configs(
            config.selection_mask_configs
        )

  def _get_mask_configs(self, mask_configs: str) -> mask_lib.MaskConfigs:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _build_mask(
      self,
      mask_configs: mask_lib.MaskConfigs,
      box: bounding_box.BoundingBoxBase,
  ) -> Any:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def output_type(self, input_type):
    return np.float32

  def subvolume_size(self):
    size = self._config.patch_size * 8
    return subvolume_processor.SuggestedXyz(size, size, 16)

  def context(self):
    pre = self._config.patch_size // 2
    post = self._config.patch_size - pre
    if self._config.fixed_current:
      if self._config.z_stride > 0:
        return (pre, pre, 0), (post, post, self._config.z_stride)
      else:
        return (pre, pre, -self._config.z_stride), (post, post, 0)
    else:
      if self._config.z_stride > 0:
        return (pre, pre, self._config.z_stride), (post, post, 0)
      else:
        return (pre, pre, 0), (post, post, -self._config.z_stride)

  def num_channels(self, input_channels):
    del input_channels
    return (
        flow_field.JAXMaskedXCorrWithStatsCalculator.non_spatial_flow_channels
        + 2
    )

  def pixelsize(self, psize):
    psize = psize.copy().astype(np.float32)
    psize[:2] *= self._config.stride
    return psize

  def process(self, subvol: Subvolume) -> SubvolumeOrMany:
    box = subvol.bbox
    input_ndarray = subvol.data
    beam_utils.counter(self.namespace, 'subvolumes-started').inc()

    assert input_ndarray.shape[0], 'Input volume should have 1 channel.'
    image = input_ndarray[0, ...]
    sel_mask = mask = None

    with beam_utils.timer_counter(self.namespace, 'build-mask'):
      if self._config.mask_configs is not None:
        mask = self._build_mask(self._config.mask_config, box)

      if self._config.selection_mask_configs is not None:
        sel_box = box.scale(
            [1.0 / self._config.stride, 1.0 / self._config.stride, 1]
        )
        sel_mask = self._build_mask(
            self._config.selection_mask_configs, sel_box
        )

    def _estimate_flow(z_prev, z_curr):
      mask_prev = mask_curr = None
      prev = image[z_prev, ...]
      curr = image[z_curr, ...]

      if mask is not None:
        mask_prev = mask[z_prev, ...]
        mask_curr = mask[z_curr, ...]

      smask = None
      if sel_mask is not None:
        smask = sel_mask[z_curr, ...]

      return mfc.flow_field(
          prev,
          curr,
          self._config.patch_size,
          self._config.stride,
          mask_prev,
          mask_curr,
          mask_only_for_patch_selection=self._config.mask_only_for_patch_selection,
          selection_mask=smask,
          batch_size=self._config.batch_size,
      )

    with beam_utils.timer_counter(self.namespace, 'flow'):
      mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
      flows = []

      if self._config.fixed_current:
        if self._config.z_stride > 0:
          rng = range(0, image.shape[0] - 1)
          z_curr = image.shape[0] - 1
        else:
          rng = range(1, image.shape[0])
          z_curr = 0
        for z_prev in rng:
          flows.append(_estimate_flow(z_prev, z_curr))
      else:
        if self._config.z_stride > 0:
          rng = range(0, image.shape[0] - self._config.z_stride)
        else:
          rng = range(-self._config.z_stride, image.shape[0])

        for z in rng:
          flows.append(_estimate_flow(z, z + self._config.z_stride))

    ret = np.array(flows)

    # Output starts at:
    #   Δz > 0: box.start.z + Δz
    #   Δz < 0: box.start.z
    out_box = self.crop_box(box)
    out_box = bounding_box.BoundingBox(
        start=out_box.start // [self._config.stride, self._config.stride, 1],
        size=[ret.shape[-1], ret.shape[-2], out_box.size[2]],
    )
    if ret.shape[0] != out_box.size[2]:
      raise ValueError(f'ret:{ret.shape} vs out:{out_box.size}')

    beam_utils.counter(self.namespace, 'subvolumes-done').inc()
    return Subvolume(np.transpose(ret, (1, 0, 2, 3)), out_box)

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


# TODO(blakely): Remove references to volinfos in favor of metadata
class ReconcileAndFilterFlows(subvolume_processor.SubvolumeProcessor):
  """Filters 4-channel or 3-channel flow volumes.

  The input flow volume(s) (generated by EstimateFlow) are filtered to
  only retain 'valid' entries fulfilling local consistency and estimation
  confidence criteria. If additional (lower-resolution) flow estimates
  are provided via 'flow_volinfos', they are used to fill any flow
  entries considered 'invalid' after filtering the higher resolution
  results.
  """

  crop_at_borders = False

  @dataclasses_json.dataclass_json
  @dataclasses.dataclass(eq=True)
  class ReconcileFlowsConfig(utils.NPDataClassJsonMixin):
    """Configuration for ReconcileAndFilterFlows.

    Attributes:
      flow_volinfos: List or comma-separated string of volinfo paths, sorted in
        ascending order of voxel size; a path can optionally be followed by
        ':scale', which defines a divisor to apply to the corresponding flow
        field. If the divisor is not specified, its value is inferred from the
        pixel size ratio between the given flow field and the first flow field
        on the list.
      mask_configs: MaskConfigs proto in text format; masked voxels will be set
        to nan (in both channels)
      min_peak_ratio: See flow_utils.clean_flow.
      min_peak_sharpness: See flow_utils.clean_flow.
      max_magnitude: See flow_utils.clean_flow.
      max_deviation: See flow_utils.clean_flow.
      max_gradient: See flow_utils.clean_flow.
      min_patch_size: See flow_utils.clean_flow.
      multi_section: If generating a multi-section volume, the value of the 3rd
        channel to initialize the output flow with
      base_delta_z: If generating a multi-section volume, the value of the 3rd
        channel to initialize the output flow with
    """

    flow_volinfos: Sequence[str] | str | None = None
    mask_configs: str | mask_lib.MaskConfigs | None = None
    min_peak_ratio: float = 1.6
    min_peak_sharpness: float = 1.6
    max_magnitude: float = 40
    max_deviation: float = 10
    max_gradient: float = 40
    min_patch_size: int = 400
    multi_section: bool = False
    base_delta_z: int = 0

  _config: ReconcileFlowsConfig

  def __init__(
      self,
      config: ReconcileFlowsConfig,
      input_volinfo_or_metadata: str | metadata.VolumeMetadata | None = None,
  ):
    """Constructor.

    Args:
      config: Parameters for ReconcileAndFilterFlows
      input_volinfo_or_metadata: input volume with a voxel size equal or smaller
        than the first volume in the flow_volinfos list
    """
    self._config = config

    self._scales = [None]
    self._metadata: list[metadata.VolumeMetadata] = []
    if input_volinfo_or_metadata is not None:
      self._metadata.append(self._get_metadata(input_volinfo_or_metadata))
    if isinstance(config.flow_volinfos, str):
      config.flow_volinfos = config.flow_volinfos.split(',')

    for path in config.flow_volinfos:
      path, _, scale = path.partition(':')
      if scale:
        scale = float(scale)
      else:
        scale = None

      self._scales.append(scale)
      self._metadata.append(self._get_metadata(path))

    # Ensure that the volumes are correctly sorted.
    for a, b in zip(self._metadata, self._metadata[1:]):
      assert a.pixel_size.x <= b.pixel_size.x
      assert a.pixel_size.y <= b.pixel_size.y
      assert a.pixel_size.x / b.pixel_size.x == a.pixel_size.y / b.pixel_size.y
      assert a.pixel_size.z == b.pixel_size.z

    if config.mask_configs is not None:
      if isinstance(config.mask_configs, str):
        config.mask_configs = self._get_mask_configs(config.mask_configs)

  def _open_volume(self, path: file.PathLike) -> base.Volume:
    """Returns a CZYX-shaped ndarray-like object."""
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _get_metadata(self, path) -> metadata.VolumeMetadata:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _get_mask_configs(self, mask_configs: str) -> mask_lib.MaskConfigs:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _build_mask(
      self,
      mask_configs: mask_lib.MaskConfigs,
      box: bounding_box.BoundingBoxBase,
  ) -> Any:
    """Returns a CZYX-shaped ndarray-like object."""
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def num_channels(self, input_channels=0):
    del input_channels
    return 2 if not self._config.multi_section else 3

  def process(self, subvol: Subvolume) -> SubvolumeOrMany:
    box = subvol.bbox
    if self._config.mask_configs is not None:
      mask = self._build_mask(self._config.mask_configs, box)
    else:
      mask = None

    # Points in image space at which the base (highest resolution) flow
    # is defined. Pixel values are assumed to correspond to the middle
    # point of the pixel.
    qy, qx = np.mgrid[: box.size[1], : box.size[0]]
    qx = qx + box.start[0]
    qy = qy + box.start[1]

    flows = []
    volumes = [self._open_volume(v) for v in self._metadata]

    for i, (vol, mag_scale) in enumerate(zip(volumes, self._scales)):
      if i > 0:
        scale = self._metadata[0].pixel_size.x / self._metadata[i].pixel_size.x
        assert scale <= 1.0
        read_box = box.scale((scale, scale, 1))
        if scale < 1:
          read_box = read_box.adjusted_by(
              start=-self._context[0], end=self._context[1]
          )
        read_box = vol.clip_box_to_volume(read_box)
        assert read_box is not None
      else:
        scale = 1
        read_box = box

      with beam_utils.timer_counter(
          'reconcile-flows', 'time-volstore-load-%d' % i
      ):
        flow = vol[read_box.to_slice4d()]

      with beam_utils.timer_counter('reconcile-flows', 'time-clean-%d' % i):
        flow = flow_utils.clean_flow(
            flow,
            self._config.min_peak_ratio,
            self._config.min_peak_sharpness,
            self._config.max_magnitude,
            self._config.max_deviation,
        )

      if i == 0 or scale == 1:
        if self._config.multi_section and flow.shape[0] != 3:
          shape = np.array(flow.shape)
          shape[0] = 3
          nflow = np.full(shape, np.nan, dtype=flow.dtype)
          nflow[:2, ...] = flow[:2, ...]
          nflow[2, ...][np.isfinite(nflow[0, ...])] = self._config.base_delta_z
          flow = nflow

        flows.append(flow)
        continue

      # Upsample flow to the base resolution.
      hires_flow = np.zeros_like(flows[0])

      oy, ox = np.ogrid[: read_box.size[1], : read_box.size[0]]
      ox = ox + read_box.start[0]
      oy = oy + read_box.start[1]
      ox = (ox / scale).ravel()
      oy = (oy / scale).ravel()

      if mag_scale is None:
        mag_scale = scale

      with beam_utils.timer_counter('reconcile-flows', 'time-upsample-%d' % i):
        for z in range(flow.shape[1]):
          rgi = interpolate.RegularGridInterpolator(
              (oy, ox), flow[0, z, ...], method='nearest', bounds_error=False
          )
          invalid_mask = np.isnan(rgi((qy, qx)))

          # We want to upsample the spatial components of the flow with
          # at least linear interpolation. Doing so with RegularGridInterpolator
          # in the presence of invalid entries (NaN) will cause the invalid
          # regions to grow beyond what 'nearest' upsampling would generate.
          # To avoid this, we use a resampling scheme with interpolation and
          # mask out invalid entries as if the field was resampled in
          # the 'nearest' interpolation mode.
          resampled = map_utils.resample_map(
              flow[:2, z : z + 1, ...], read_box, box, 1 / scale, 1  #
          )
          hires_flow[:2, z : z + 1, ...] = resampled / mag_scale
          hires_flow[0, z, ...][invalid_mask] = np.nan
          hires_flow[1, z, ...][invalid_mask] = np.nan

          for c in range(2, self.num_channels()):
            rgi = interpolate.RegularGridInterpolator(
                (oy, ox), flow[c, z, ...], method='nearest', bounds_error=False
            )
            hires_flow[c, z, ...] = rgi((qy, qx)).astype(np.float32)

      if mask is not None:
        flow_utils.apply_mask(hires_flow, mask)
      flows.append(hires_flow)

    ret = flow_utils.reconcile_flows(
        flows,
        self._config.max_gradient,
        self._config.max_deviation,
        self._config.min_patch_size,
    )
    return self.crop_box_and_data(box, ret)


class EstimateMissingFlow(subvolume_processor.SubvolumeProcessor):
  """Estimates a multi-section flow field.

  Takes an existing single-section (2-channel) flow volume as input,
  and tries to compute flow vectors which are invalid in the input (NaNs).
  """

  @dataclasses_json.dataclass_json
  @dataclasses.dataclass(frozen=True)
  class EstimateMissingFlowConfig:
    """Configuration for EstimateMissingFlow.

    Attributes:
      patch_size: Patch size in pixels, divisible by 'stride'
      stride: XY stride size in pixels
      delta_z: Z stride size in pixels (Δz) for the input volume
      max_delta_z: Maximum Z stride with which try to estimate missing flow
        vectors
      max_attempts: Maximum number of attempts to estimate a flow vector for an
        unmasked location
      mask_configs: MaskConfigs proto in text format specifying a mask to
        exclude some voxels from the flow calculation
      mask_only_for_patch_selection: Whether to only use mask to decide for
        which patch pairs to compute flow
      selection_mask_configs: MaskConfigs in text format specifying a mask the
        positive entries of which indicate locations for which flow should be
        computed; this mask should have the same resolution and geometry as the
        output flow volume
      min_peak_ratio: Quality threshold for acceptance of newly estimated flow
        vectors; see flow_utils.clean_flow
      min_peak_sharpness: Quality threshold for acceptance of newly estimated
        flow vectors; see flow_utils.clean_flow
      max_magnitude: Maximum magnitude of a flow vector; see
        flow_utils.clean_flow
      batch_size: Max number of patches to process in parallel
      image_volinfo: Path to the VolumeInfo descriptor of the image volume
      image_cache_bytes: Number of bytes to use for the in-memory image cache;
        this should ideally be large enough so that no chunks are loaded more
        than once when processing a subvolume
      mask_cache_bytes: Number of bytes to use for the in-memory mask cache
      search_radius: Additional radius to extend patch_size by in every
        direction when extracting data for the 'previous' section
    """

    patch_size: int = 160
    stride: int = 40
    delta_z: int = 1
    max_delta_z: int = 4
    max_attempts: int = 2
    mask_configs: str | mask_lib.MaskConfigs | None = None
    mask_only_for_patch_selection: bool = True
    selection_mask_configs: str | mask_lib.MaskConfigs | None = None
    min_peak_ratio: float = 1.6
    min_peak_sharpness: float = 1.6
    max_magnitude: int = 40
    batch_size: int = 1024
    image_volinfo: str | None = None
    image_cache_bytes: int = int(1e9)
    mask_cache_bytes: int = int(1e9)
    search_radius: int = 0

  _config: EstimateMissingFlowConfig

  def __init__(
      self,
      config: EstimateMissingFlowConfig,
      input_volinfo_or_ts_spec=None,
  ):
    """Constructor.

    Args:
      config: Parameters for EstimateMissingFlow
      input_volinfo_or_ts_spec: unused
    """
    del input_volinfo_or_ts_spec

    self._config = config

    if config.patch_size % config.stride != 0:
      raise ValueError(
          f'patch_size {config.patch_size} not a multiple of stride'
          f' {config.stride}'
      )

    self._search_patch_size = config.patch_size + config.search_radius * 2
    if self._search_patch_size % config.stride != 0:
      raise ValueError(
          f'search_patch_size {self._search_patch_size} not a multiple of'
          f' stride {config.stride}'
      )

    if config.mask_configs is not None:
      config.mask_configs = self._get_mask_configs(config.mask_configs)

    if config.selection_mask_configs is not None:
      config.selection_mask_configs = self._get_mask_configs(
          config.selection_mask_configs
      )

  def _get_mask_configs(self, mask_configs: str) -> mask_lib.MaskConfigs:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _open_volume(self, path: file.PathLike) -> base.Volume:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def _build_mask(
      self,
      mask_configs: mask_lib.MaskConfigs,
      # TODO(blakely): Switch to BoundingBox after move to 3p.
      box: bounding_box.BoundingBoxBase,
  ) -> Any:
    """Returns a CZYX-shaped ndarray-like object."""
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def num_channels(self, input_channels):
    """Returns the number of channels in the output volume.

    Args:
      input_channels: The number of channels in the input volume.

    Returned channels are `flow_x, flow_y, lookback_z`. The latter represents
    how far back in the stack the processor had to look to find a valid flow
    calculation.
    """
    del input_channels
    return 3

  def process(self, subvol: Subvolume) -> SubvolumeOrMany:
    box = subvol.bbox
    input_ndarray = subvol.data
    namespace = 'estimate-missing-flow'
    beam_utils.counter(namespace, 'subvolumes-started').inc()

    image_volume = self._open_volume(self._config.image_volinfo)

    # Bounding box identifying the region of the image for which the input
    # flow was computed.
    stride = self._config.stride
    full_image_box = bounding_box.BoundingBox(
        start=(
            box.start[0] * stride - self._search_patch_size // 2,
            box.start[1] * stride - self._search_patch_size // 2,
            box.start[2],
        ),
        size=(
            (box.size[0] - 1) * stride + self._search_patch_size,
            (box.size[1] - 1) * stride + self._search_patch_size,
            1,
        ),
    )
    prev_image_box = image_volume.clip_box_to_volume(full_image_box)
    assert prev_image_box is not None

    # Nothing to do if we don't have sufficient image context for any
    # flow field entries.
    if np.any(prev_image_box.size[:2] <= self._search_patch_size):
      return subvol

    # Do not process flow field entries for which we do not have sufficient
    # image context.
    offset = prev_image_box.translate(-full_image_box.start).start // stride
    out_box = box.adjusted_by(start=offset)
    input_ndarray = input_ndarray[:, :, offset[1] :, offset[0] :]

    # ceil_div
    offset = -((prev_image_box.end - full_image_box.end) // stride)
    out_box = out_box.adjusted_by(end=-offset)
    input_ndarray = input_ndarray[:, :, : out_box.size[1], : out_box.size[0]]

    patch_size = self._config.patch_size
    curr_image_box = bounding_box.BoundingBox(
        start=(
            out_box.start[0] * stride - patch_size // 2,
            out_box.start[1] * stride - patch_size // 2,
            out_box.start[2],
        ),
        size=(
            (out_box.size[0] - 1) * stride + patch_size,
            (out_box.size[1] - 1) * stride + patch_size,
            1,
        ),
    )
    curr_image_box = image_volume.clip_box_to_volume(curr_image_box)
    assert curr_image_box is not None

    # The input flow forms the initial state of the output. We will try
    # to fill-in any invalid (NaN) pixels by computing flow against
    # earlier sections.
    ret = np.zeros([3] + list(out_box.size[::-1]))
    ret[:2, ...] = input_ndarray
    ret[2, ...] = self._config.delta_z

    sel_mask = None
    if self._config.selection_mask_config is not None:
      sel_mask = self._build_mask(self._config.selection_mask_config, out_box)

    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    invalid = np.isnan(input_ndarray[0, ...])
    for z in range(0, invalid.shape[0]):
      z0 = box.start[2] + z
      logging.info('Processing rel_z=%d abs_z=%d', z, z0)

      if np.all(~invalid[z, ...]):
        beam_utils.counter(namespace, 'sections-already-valid').inc()
        continue

      image_box = curr_image_box.translate([0, 0, z])
      curr_mask = None
      if self._config.mask_configs is not None:
        curr_mask = self._build_mask(
            self._config.mask_configs, image_box
        ).squeeze()
        if np.all(curr_mask):
          beam_utils.counter(namespace, 'sections-masked').inc()
          continue

        logging.info('Mask built.')

      attempts = np.zeros(ret.shape[2:], dtype=int)
      mask = ~np.isfinite(ret[0, z, ...])
      if sel_mask is not None:
        mask &= sel_mask[z, ...]

      curr = image_volume.asarray[image_box.to_slice4d()].squeeze()

      delta_z = self._config.delta_z
      if delta_z > 0:
        rng = range(delta_z + 1, self._config.max_delta_z + 1)
      else:
        rng = range(delta_z - 1, self._config.max_delta_z - 1, -1)

      for delta_z in rng:
        if (
            box.start[2] - delta_z < 0
            or box.end[2] - delta_z >= image_volume.volume_size[2]
        ):
          break

        t_start = time.time()
        prev_box = prev_image_box.translate([0, 0, z - delta_z])
        logging.info('Trying delta_z=%d (%r)', delta_z, prev_box)
        prev = image_volume.asarray[prev_box.to_slice4d()].squeeze()
        logging.info('.. image loaded.')
        t1 = time.time()

        if self._config.mask_configs is not None:
          prev_mask = self._build_mask(
              self._config.mask_configs, prev_box
          ).squeeze()
          if np.all(prev_mask):
            continue
        else:
          prev_mask = None
        logging.info('.. mask loaded.')

        # Limit the number of estimation attempts per voxel. Attempts
        # are only counted when voxels in both sections are unmasked.
        mask &= attempts <= self._config.max_attempts
        if not np.any(mask):
          break

        logging.info('.. points to evaluate: %d', np.sum(mask))
        t2 = time.time()

        flow = mfc.flow_field(
            prev,
            curr,
            self._search_patch_size,
            self._config.stride,
            prev_mask,
            curr_mask,
            mask_only_for_patch_selection=self._config.mask_only_for_patch_selection,
            selection_mask=mask,
            batch_size=self._config.batch_size,
            post_patch_size=self._config.patch_size,
        )

        t3 = time.time()
        valid = np.isfinite(flow[0, ...])
        attempts[: valid.shape[0], : valid.shape[1]][valid] += 1

        flow = flow_utils.clean_flow(
            flow[:, np.newaxis, ...],  #
            self._config.min_peak_ratio,
            self._config.min_peak_sharpness,
            self._config.max_magnitude,
            max_deviation=0.0,
        )

        t4 = time.time()
        sy, sx = flow.shape[2:]
        to_update = mask[:sy, :sx] & np.isfinite(flow[0, 0, ...])
        mask[:sy, :sx][to_update] = False
        logging.info('.. points to update: %d', np.sum(to_update))

        beam_utils.counter(namespace, f'sections-filled-delta{delta_z}').inc(
            np.sum(to_update)
        )
        ret[2, z, :sy, :sx][to_update] = delta_z
        ret[0, z, :sy, :sx][to_update] = flow[0, 0, ...][to_update]
        ret[1, z, :sy, :sx][to_update] = flow[1, 0, ...][to_update]
        t5 = time.time()

        logging.info(
            'timings: img:%.2f  mask:%.2f  flow:%.2f  clean:%.2f  update:%.2f',
            t1 - t_start,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4,
        )

    return Subvolume(ret, out_box)
