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
"""Processors for coordinate maps."""

import bisect
import dataclasses
import functools

from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.volume import metadata
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import dataclasses_json
import numpy as np
from scipy import spatial
from sofima import map_utils

Subvolume = subvolume.Subvolume
SubvolumeOrMany = subvolume_processor.SubvolumeOrMany


class ReconcileCrossBlockMaps(subvolume_processor.SubvolumeProcessor):
  """Reconciles three coordinate maps with different Z resolutions.

  The three volumes taken as input are:
    - (main): high res. coordinate map (computed for non-overlapping
      blocks)
    - (last): high res. coordinate map covering only the first section
      of every block, and representing its position as if it was part
      of the preceding block
    - (cross_block): lower res. coordinate map (higher Z stride)

  The typical use case is to have a block-wise optimized mesh
  as the main input volume, and a cross-block optimized mesh as the
  cross_block volume. That volume then provides the final position of the nodes
  of the first section of every block, and the node positions in the block
  interior are interpolated to fit these. This minimally perturbs the
  section-to-section alignment, while generating a globally contiguous
  geometry.
  """

  @dataclasses.dataclass(eq=True)
  class Config(dataclasses_json.DataClassJsonMixin):
    """Configuration for ReconcileCrossBlockMaps.

    Attributes:
      cross_block: path to the low-res, cross-block coordinate map volume
      cross_block_inv: the inverse of the map in cross_block_path
      last_inv: path to the inverse of the coordinate map providing the position
        of the first section of every block as if it was part of the previous
        block
      main_inv: the inverse of the map used as processor input (only the last
        section within the volume is used)
      z_map: dictionary mapping coordinates of the high-res map to the low-res
        map
      stride: pixel distance between nearest neighbors of the coordinate maps,
        in pixels of the output volume
      xy_overlap: neighboring subvolume overlap in the XY directions, in units
        of pixels of main input volume
      backward: whether the mesh was solved in backward mode (proceeding from
        higher z coordinates towards lower ones)
    """

    cross_block: metadata.DecoratedVolume | str
    cross_block_inv: metadata.DecoratedVolume | str
    last_inv: metadata.DecoratedVolume | str
    main_inv: metadata.DecoratedVolume | str
    z_map: dict[str, int]
    stride: int
    xy_overlap: int = 128
    backward: bool = False

  _config: Config

  crop_at_borders = False

  def __init__(
      self,
      config: Config,
      input_volinfo=None,
  ):
    """Constructor.

    Args:
      config: parameters for ReconcileCrossBlockMaps
      input_volinfo: path to the high-res input volume (unused)
    """
    del input_volinfo
    self._config = config
    self._main_inv_volinfo = config.main_inv
    self._xblock_volinfo = config.cross_block
    self._xblock_inv_volinfo = config.cross_block_inv
    self._last_inv_volinfo = config.last_inv
    self._xy_overlap = config.xy_overlap
    self._z_map = {int(k): int(v) for k, v in config.z_map.items()}
    self._sorted_z = list(sorted(self._z_map.keys()))
    self._stride = config.stride
    self._backward = config.backward

  def context(self):
    pre = self._xy_overlap // 2
    post = self._xy_overlap - pre
    return (pre, pre, 1), (post, post, 0)

  def _get_z_range(self, z: int):
    """Returns the first and last + 1 sections of the block for 'z'."""
    idx = bisect.bisect_left(self._sorted_z, z)
    if idx == 0:
      return 0, self._sorted_z[idx]
    else:
      return self._sorted_z[idx - 1], self._sorted_z[idx]

  def _interpolate(
      self,
      data: np.ndarray,
      box: bounding_box.BoundingBox,
      z0: int,
      z1: int,
      load_main_inv,
      load_last_inv,
      load_xblock,
      load_xblock_inv,
      done: set[int],
  ):
    """Interpolates in-block coordinate map entries.

    The data is interpolated so that the map entries in the first section
    of every block are equal to their values in the 'cross_block' volume,
    and so that section-to-section distortion is minimized within the block,
    relative to the positions originally present in 'data'.

    Updates 'done' with any processed sections.

    Args:
      data: [2, z, y, x] coordinate map to adjust
      box: box from which 'data' was extracted
      z0: first section of the current block
      z1: first section of the next block
      load_main_inv: callable to load a section from the inverse of the 'main'
        volume
      load_last_inv: callable to load a section from the inverse of the 'last'
        volume
      load_xblock: callable to load a section from the 'cross_block' volume
      load_xblock_inv: callable to load a section from the inverse of the
        'cross_block' volume
      done: set of 'z' section coordinates that have already been processed
    """
    if self._backward:
      xblock_post = load_xblock(self._z_map[z0])
    else:
      xblock_post = load_xblock(self._z_map[z1])

    if not self._backward and z0 > 0:
      xblock_pre = load_xblock(self._z_map[z0])
      xblock_pre_inv = load_xblock_inv(self._z_map[z0])
    elif self._backward and z1 < self._sorted_z[-1]:
      xblock_pre = load_xblock(self._z_map[z1])
      xblock_pre_inv = load_xblock_inv(self._z_map[z1])
    else:
      xblock_pre_inv = xblock_pre = np.zeros_like(xblock_post)

    if self._backward:
      if z0 != self._sorted_z[0]:
        block_end_inv = load_last_inv(z0)
      else:
        block_end_inv = load_main_inv(z0)
    else:
      if z1 != self._sorted_z[-1]:
        block_end_inv = load_last_inv(z1)
      else:
        block_end_inv = load_main_inv(z1)

    flat_box = bounding_box.BoundingBox(
        start=box.start, size=(box.size[0], box.size[1], 1)
    )

    # The interpolation is done so that the first section of the block ends up
    # at 'xblock_pre', the first section of following block at 'xblock_post',
    # and the in-block relative movement of the last section of the block
    # ('block_end') is eliminated in favor of 'xblock_post'.
    #
    # To do so, we compute an offset field between 'block_end' and
    # 'xblock_post'. This field, scaled by the relative vertical position
    # within the block (z/b), can be used to warp the *aligned* sections within
    # the block to achieve the desired final result:
    #
    #   warp(warp(unaligned_image, section * xblock_pre)^-1), scaled_offset^-1)
    #
    # which is equivalent to:
    #
    #   warp(unaligned_image, scaled_offset^-1 * (section * xblock_pre)^-1)
    #
    # where * denotes coordinate map composition.
    #
    # Using the known desired final position of the last section of the block:
    #
    #   xblock_post = (block_end * xblock_pre) * offset
    #   offset = (block_end * xblock_pre)^-1 * xblock_post
    #   offset = (xblock_pre^-1 * block_end^-1) * xblock_post
    offset = map_utils.compose_maps(
        map_utils.compose_maps(
            xblock_pre_inv,
            flat_box,
            self._stride,
            block_end_inv,
            flat_box,
            self._stride,
        ),
        flat_box,
        self._stride,
        xblock_post,
        flat_box,
        self._stride,
    )

    block_size = z1 - z0
    for z in range(max(box.start[2], z0), min(box.end[2], z1 + 1)):
      i = z - z0
      # Each section can be processed only once.
      if z in done:
        continue
      rel_z = z - box.start[2]

      if i == block_size:
        data[:, rel_z : rel_z + 1, ...] = (
            xblock_pre if self._backward else xblock_post
        )
      elif i == 0:
        data[:, rel_z : rel_z + 1, ...] = (
            xblock_post if self._backward else xblock_pre
        )
      else:

        if self._backward:
          scale = (block_size - i) / block_size
        else:
          scale = i / block_size

        try:
          # The output coordinate map here is the inverse of the argument
          # passed to warp() in the comment above, i.e.:
          #   (section * xblock_pre) * scaled_offset
          interior_aligned = map_utils.compose_maps(
              data[:, rel_z : rel_z + 1, ...],  #
              flat_box,
              self._stride,
              xblock_pre,
              flat_box,
              self._stride,
          )
          data[:, rel_z : rel_z + 1, ...] = map_utils.compose_maps(  #
              interior_aligned,
              flat_box,
              self._stride,
              offset * scale,
              flat_box,
              self._stride,
          )
        except spatial.qhull.QhullError:
          pass

      done.add(z)

  def process(self, subvol: Subvolume) -> SubvolumeOrMany:
    box = subvol.bbox
    coord_map = subvol.data
    xblock_volstore = self._open_volume(self._xblock_volinfo)
    xblock_inv_volstore = self._open_volume(self._xblock_inv_volinfo)
    last_inv_volstore = self._open_volume(self._last_inv_volinfo)
    main_inv_volstore = self._open_volume(self._main_inv_volinfo)

    def _load_section(z, volstore):
      load_box = bounding_box.BoundingBox(
          start=(box.start[0], box.start[1], z),
          size=(box.size[0], box.size[1], 1),
      )
      return volstore[load_box.to_slice4d()]

    load_main_inv = functools.partial(_load_section, volstore=main_inv_volstore)
    load_last_inv = functools.partial(_load_section, volstore=last_inv_volstore)
    load_xblock = functools.partial(_load_section, volstore=xblock_volstore)
    load_xblock_inv = functools.partial(
        _load_section, volstore=xblock_inv_volstore
    )

    ranges = []
    z = box.start[2]
    while z < box.end[2]:
      s, e = self._get_z_range(z)
      ranges.append((s, e))
      z = e + 1

    ret = coord_map.copy()
    done = set()
    # Interpolate coord_map blockwise.
    for s, e in ranges:
      self._interpolate(
          ret,
          box,
          s,
          e,
          load_main_inv,
          load_last_inv,
          load_xblock,
          load_xblock_inv,
          done,
      )

    # Check that all sections have been processed.
    assert not set(range(box.start[2], box.end[2])) - done

    ret[np.isnan(coord_map)] = np.nan
    return self.crop_box_and_data(box, ret)


class InvertMap(subvolume_processor.SubvolumeProcessor):
  """Inverts a coordinate map."""

  @dataclasses.dataclass(eq=True)
  class Config(dataclasses_json.DataClassJsonMixin):
    """Configuration for coordinate map inversion.

    Attributes:
      stride: [Z]YX stride of the coordinate map
      crop_output: if False, outputs data for the input box instead of the inner
        box of the map; a typical use case is when inverting data for a complete
        section in which case there are no other work items that could provide
        data for areas outside of the inner box
      input_volume: path to the volume containing the coordinate map to invert.
    """

    stride: map_utils.StrideZYX
    crop_output: bool = True
    # TODO(blakely): Pass this in via the constructor.
    input_volume: str | None = None

  _config: Config
  crop_at_borders = False
  output_num = subvolume_processor.OutputNums.MULTI

  def __init__(
      self,
      config: Config,
      input_path_or_metadata: (
          file.PathLike | metadata.VolumeMetadata | None
      ) = None,
  ):

    source_volume = input_path_or_metadata
    if source_volume is None:
      source_volume = config.input_volume
    if source_volume is None:
      raise ValueError('No source volume specified.')

    self._config = config
    meta = self._get_metadata(source_volume)

    self._volume_bbox = bounding_box.BoundingBox(
        start=(0, 0, 0),
        size=(meta.volume_size.x, meta.volume_size.y, meta.volume_size.z),
    )

  def process(self, subvol: Subvolume) -> SubvolumeOrMany:
    config = self._config
    box = subvol.bbox
    input_ndarray = subvol.data
    # If the map is completely invalid, there is nothing to invert.
    if np.all(np.isnan(input_ndarray)):
      return []

    rel_map = input_ndarray.astype(np.float64)

    if config.crop_output:
      dst_box = map_utils.inner_box(rel_map, box, config.stride)
      dst_box = dst_box.intersection(self._volume_bbox)
    else:
      dst_box = box

    if dst_box is None:
      return []

    inv_map = map_utils.invert_map(rel_map, box, dst_box, config.stride)
    return [Subvolume(inv_map, dst_box)]


class ResampleMap(subvolume_processor.SubvolumeProcessor):
  """Resamples a coordinate map."""

  @dataclasses.dataclass(eq=True)
  class Config(dataclasses_json.DataClassJsonMixin):
    """Configuration for map resampling.

    Attributes:
    """

    stride: int
    out_stride: int
    scale: float = 1.0
    method: str = 'linear'

  crop_at_borders = False
  output_num = subvolume_processor.OutputNums.MULTI

  _config: Config

  def __init__(self, config: Config, input_volinfo_or_ts=None):
    del input_volinfo_or_ts
    self._config = config

  def pixelsize(self, psize):
    psize = psize.copy().astype(np.float32)
    psize[:2] *= self._config.out_stride / self._config.stride
    return psize

  def process(self, subvol: Subvolume) -> SubvolumeOrMany:
    config = self._config
    box = subvol.bbox
    input_ndarray = subvol.data
    if np.all(np.isnan(input_ndarray)):
      return []

    rel_map = input_ndarray.astype(np.float64) * config.scale
    dst_box = self.crop_box(box)
    ratio = config.stride / config.out_stride
    dst_box = dst_box.scale([ratio, ratio, 1.0])

    out_map = map_utils.resample_map(
        rel_map, box, dst_box, config.stride, config.out_stride, config.method
    )

    return [Subvolume(out_map, dst_box)]


class MaskIrregularities(subvolume_processor.SubvolumeProcessor):
  """Masks irregulariaties in a coordinate map volume."""

  crop_at_borders = False

  def __init__(self, stride, frac, input_volinfo=None):
    self._stride = stride
    self._frac = frac
    del input_volinfo

  def context(self):
    # XY context sufficient to account for the dilation applied in
    # 'mask_irregular'.
    return (3, 3, 0), (3, 3, 0)

  def process(self, subvol: Subvolume) -> SubvolumeOrMany:
    box = subvol.bbox
    input_ndarray = subvol.data
    ret = np.zeros_like(input_ndarray)

    for z in range(0, input_ndarray.shape[1]):
      section = input_ndarray[:, z, ...].copy()
      map_utils.mask_irregular(section, self._stride, self._frac)
      ret[:, z, ...] = section

    return self.crop_box_and_data(box, ret)


class FillMissing(subvolume_processor.SubvolumeProcessor):
  """Fills missing entries in a coordinate map by inter/extrapolation."""

  @dataclasses.dataclass(eq=True)
  class Config(dataclasses_json.DataClassJsonMixin):
    """Configuration for filling in missing entries in a coordinate map.

    Currently empty, but required by the processing framework.
    """

  _config: Config

  crop_at_borders = False

  def __init__(self, input_volinfo=None):
    del input_volinfo

  def process(self, subvol: Subvolume) -> SubvolumeOrMany:
    box = subvol.bbox
    mesh = subvol.data
    if not np.all(np.isnan(mesh)):
      mesh = map_utils.fill_missing(mesh, extrapolate=True)

    return self.crop_box_and_data(box, mesh)
