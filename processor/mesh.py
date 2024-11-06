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

"""Mesh processors for SOFIMA."""

import dataclasses
import enum
from typing import Sequence

from absl import logging
from connectomics.common import bounding_box
from connectomics.common import file
from connectomics.common import utils
from connectomics.volume import mask as mask_lib
from connectomics.volume import metadata
from connectomics.volume import subvolume
from connectomics.volume import subvolume_processor
import dataclasses_json
import numpy as np
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh as mesh_lib
from sofima.processor import client_utils


Subvolume = subvolume.Subvolume


class SolutionStatus(enum.IntEnum):
  UNDEFINED = -1
  REGULAR = 0
  PREP_FAILED = 1
  REGULARIZED = 2


class MeshInitState(enum.Enum):
  ZEROS = 0
  PREV_MEDIAN = 1


@dataclasses.dataclass(frozen=True)
class FlowVolume(dataclasses_json.DataClassJsonMixin):
  """Flow volume."""

  delta_z: int
  volume: metadata.DecoratedVolume


@dataclasses.dataclass(frozen=True)
class BadSectionRange(dataclasses_json.DataClassJsonMixin):
  """Bad section range."""

  start: int
  end: int
  # Volume defining the flow between the 1st section after the range,
  # and the last section before the range.
  #
  # For forward optimization:
  #   the value at z = end + 1 should contain flow estimates between
  #   end + 1 (post) and start - 1 (pre).
  #
  # For backward optimization:
  #   the value at z = start - 1 should contain flow estimates between
  #   start - 1 (post) and end + 1 (pre)
  flow: FlowVolume


@dataclasses.dataclass(frozen=True)
class MeshOptions(dataclasses_json.DataClassJsonMixin):
  init_state: MeshInitState = MeshInitState.ZEROS
  irregular_mask_radius: int | None = None


# Describes the optimization settings to apply to the first complete
# section after a coming-in region within which sections grow progressively
# smaller (marked with 'x'):
#
# --------------    <- last complete section
# ------------   x  <- first partial section
# --------       x
# ----           x  <- last partial section
# --------------    <- first complete section (z)
#
# The last complete section is optimized with a custom 3-channel flow
# volume, where the 3rd channel indicates a relative delta_z offset
# to the section against which the flow was computed (generally the
# last section in the coming-in region with matching image content).
@dataclasses.dataclass(frozen=True)
class ComingIn(dataclasses_json.DataClassJsonMixin):
  z: int
  flow: metadata.DecoratedVolume


# TODO(blakely): Reenable counters that work both internally and externally.
class RelaxMesh(subvolume_processor.SubvolumeProcessor):
  """Performs mesh relaxation."""

  @dataclasses.dataclass(eq=True)
  class Config(utils.NPDataClassJsonMixin):
    """Configuration for mesh relaxation.

    Attributes:
      output_dir: Directory to write output to
      integration_config: Solver configuration for mesh relaxation.
      mesh: Initial state of the mesh. When not specified, starts from an empty
        mesh.
      flows: Inter-section optical flow.
      sections_to_skip: Sections that should be skipped. These sections do not
        contribute to the global shape. They are optimized normally, but cannot
        serve as a reference section when optimizing non-skipped sections. When
        these sections are used, a delta_z=2 volume has to be provided so that
        the following section can be aligned to the preceding one.
      ranges_to_skip: Ranges of at least 2 consecutive sections to be skipped
        (same as in the previous field).
      mask: Mask for invalid tissue. Corresponding nodes will be removed from
        the mesh. Folds are ignored in masked areas.
      block_starts: Section numbers for which no preceding solved section is
        needed. The last solved z coordinate is checked against this list.
      block_ends: Last sections to be optimized within a block. Alignment is not
        done across blocks.
      backward: If true, optimization is done towards lower z values. Expects
        all flows to be computed with negative delta_z.
      mesh_min_frac: Min/max horizontal/vertical separation of nearest neighbor
        mesh nodes, as a fraction of integration_config.stride. Mesh nodes
        violating these conditions will be considered invalid, and their
        positions replaced with nan's when used in a reference section.
      mesh_max_frac: Max horizontal/vertical separation
      coming_in: Coming-in region settings. One entry per region.
      options: Mesh options.
    """
    output_dir: str
    integration_config: mesh_lib.IntegrationConfig
    mesh: metadata.DecoratedVolume | None
    flows: list[FlowVolume]
    sections_to_skip: list[int]
    ranges_to_skip: list[BadSectionRange]
    mask: str | mask_lib.MaskConfigs | None
    block_starts: list[int]
    block_ends: list[int]
    backward: bool
    mesh_min_frac: float
    mesh_max_frac: float
    coming_in: list[ComingIn]
    options: MeshOptions | None = dataclasses.field(default_factory=MeshOptions)

  _config: Config

  def __init__(self, config: Config, input_ts_spec=None):
    self._config = config

  def _load_stitched_tile(
      self, output_dir: file.PathLike, box: bounding_box.BoundingBoxBase
  ) -> np.ndarray | None:
    raise NotImplementedError(
        'This function needs to be defined in a subclass.'
    )

  def compute_ref_mesh_multiz(
      self,
      flow: np.ndarray,
      box: bounding_box.BoundingBoxBase,
      starts: Sequence[int],
      stride: Sequence[float],
      ignore_xblock: bool = True,
      allow_missing_mesh: bool = True,
  ) -> np.ndarray:
    """Computes the mesh state of a reference section given a multi-section flow."""
    config = self._config
    z_offsets = np.unique(flow[2, 0, :, :])
    z_offsets = z_offsets[np.isfinite(z_offsets) & (z_offsets != 0)]
    z_offsets = z_offsets.astype(np.int32).tolist()
    mesh_state = np.full([2] + list(flow.shape[1:]), np.nan)

    z = box.start[2]
    curr_block = client_utils.get_block_id(z, starts, config.backward)
    for delta_z in sorted(z_offsets, key=abs):
      ref_block = client_utils.get_block_id(
          z - delta_z, starts, config.backward
      )
      if curr_block != ref_block:
        if ignore_xblock:
          break
        else:
          raise ValueError(
              'Mesh data needs to be within a single block '
              f'({z} vs {z - delta_z}.'
          )

      offset = np.array([0, 0, delta_z])
      ref_box = box.translate(-offset)
      logging.info('Attempting to load ref. mesh for %r', ref_box)
      ref_mesh = self._load_stitched_tile(config.output_dir, ref_box)
      if ref_mesh is None:
        if allow_missing_mesh:
          assert config.mesh is not None
          ref_volume = self._open_volume(config.mesh)
          ref_mesh = ref_volume[ref_box.to_slice4d()]
        else:
          raise ValueError(f'Missing previous mesh data for {ref_box.start}')

      if config.mask is not None:
        mask = self._build_mask(config.mask, ref_box)
        flow_utils.apply_mask(ref_mesh, mask)

      m = flow[2, ...] == delta_z
      curr_flow = flow[:2, ...].copy()
      curr_flow[0, ...][~m] = np.nan
      curr_flow[1, ...][~m] = np.nan

      curr_flow = np.array(
          map_utils.compose_maps_fast(  # pytype: disable=wrong-arg-types  # jax-ndarray
              curr_flow,
              box.start[::-1],
              stride,
              ref_mesh,
              box.start[::-1],
              stride,
          )
      )

      mesh_state[0, ...][m] = curr_flow[0, ...][m]
      mesh_state[1, ...][m] = curr_flow[1, ...][m]

    return mesh_state

  def is_skipped_section(self, z: int) -> bool:
    if z in self._config.sections_to_skip:
      return True

    for rng in self._config.ranges_to_skip:
      if z >= rng.start and z <= rng.end:
        return True

    return False

  def compute_ref_mesh(
      self,
      flow: np.ndarray,
      ref_box: bounding_box.BoundingBoxBase,
      stride: Sequence[float],
  ) -> np.ndarray:
    """Computes the mesh state of a reference section."""
    config = self._config
    ref_mesh = self._load_stitched_tile(config.output_dir, ref_box)
    if ref_mesh is None:
      assert config.mesh is not None
      ref_volume = self._open_volume(config.mesh)
      ref_mesh = ref_volume[ref_box.to_slice4d()]

    if config.mesh is not None:
      mask = self._build_mask(config.mask, ref_box)
      flow_utils.apply_mask(ref_mesh, mask)

    flow = np.array(
        map_utils.compose_maps_fast(  # pytype: disable=wrong-arg-types  # jax-ndarray
            flow,
            ref_box.start[::-1],
            stride,
            ref_mesh,
            ref_box.start[::-1],
            stride,
        )
    )

    return flow

  def get_prev_state(
      self, stride: Sequence[float], bbox: bounding_box.BoundingBox
  ) -> np.ndarray | None:
    """Computes the positions of reference nodes for a section.

    If flows are computed against more than one section, they are averaged
    together. This works because of linearity of Hooke's law.

    Args:
      stride: distance between nearest neighbors of the mesh in pixels
      bbox: bounding box of the section to be optimized

    Returns:
      positions of the reference nodes in relative format
    """
    config = self._config

    z = bbox.start[2]
    starts = list(sorted(config.block_starts))
    if z in starts:
      # The first section of the block is not optimized and does not
      # require any context state.
      return

    # The first section after a coming-in region requires special handling.
    for cin in config.coming_in:
      if z != cin.z:
        continue

      flow_volume = self._open_volume(cin.flow)
      flow: np.ndarray = flow_volume[bbox.to_slice4d()]
      return self.compute_ref_mesh_multiz(
          flow,
          bbox,
          starts,
          stride,
          # load_kwargs,
          ignore_xblock=False,
          allow_missing_mesh=False,
      )

    flows = config.flows

    # If the previous section is the last one in a skipped range, use a custom
    # flow volume as specified in the config.
    prev_z = z - (-1 if config.backward else 1)
    for rng in config.ranges_to_skip:
      if prev_z == rng.end:
        flows = [rng.flow]
        break

    curr_block = client_utils.get_block_id(z, starts, config.backward)
    prev = np.zeros((2, 1, bbox.size[1], bbox.size[0]))
    count = np.zeros((bbox.size[1], bbox.size[0]), dtype=np.int32)
    num_refs = 0
    for flow in flows:
      # Do not try to to align to skipped sections.
      ref_z = z - flow.delta_z
      if self.is_skipped_section(ref_z):
        continue

      # Only consider previous sections from the current block.
      ref_block = client_utils.get_block_id(ref_z, starts, config.backward)
      if ref_block != curr_block:
        continue

      flow_volume = self._open_volume(flow.volume)

      flow_field = flow_volume[bbox.to_slice4d()]
      if flow_volume.info.num_channels == 2:
        offset = np.array([0, 0, flow.delta_z])
        ref_box = bbox.translate(-offset)
        ref_mesh = self.compute_ref_mesh(flow_field, ref_box, stride)
      else:
        ref_mesh = self.compute_ref_mesh_multiz(
            flow_field,
            bbox,
            starts,
            stride,
        )

      count += np.isfinite(ref_mesh[0, 0, ...]).astype(np.int32)
      np.nan_to_num(ref_mesh, copy=False)

      prev += ref_mesh
      num_refs += 1

    if num_refs == 0:
      return

    count = count.astype(np.float32)
    count[count == 0] = np.nan
    prev = prev / count[np.newaxis, np.newaxis, :, :]

    mask_radius = 1
    if config.options and config.options.irregular_mask_radius is not None:
      mask_radius = config.options.irregular_mask_radius

    map_utils.mask_irregular(
        prev[:, 0, ...],
        stride,
        config.mesh_min_frac,
        config.mesh_max_frac,
        dilation_iters=mask_radius,
    )

    return prev

  def maybe_update_init_state(
      self,
      x: np.ndarray,
      prev: np.ndarray | None,
      options: MeshOptions,
  ) -> np.ndarray:
    if options.init_state == MeshInitState.PREV_MEDIAN and prev is not None:
      x[0, ...] = np.nanmedian(prev[0, ...])
      x[1, ...] = np.nanmedian(prev[1, ...])
      x = np.nan_to_num(x)

    return x

  def get_mesh_state(
      self,
      box: bounding_box.BoundingBoxBase,
      stride: Sequence[float],
      prev: np.ndarray | None,
  ) -> np.ndarray:
    """Returns the state of the section to be optimized."""
    config = self._config
    if config.mesh is None:
      return np.zeros((2, 1, box.size[1], box.size[0]))

    mesh_volume = self._open_volume(config.mesh)
    state = mesh_volume[box.to_slice4d()]

    masked = map_utils.mask_irregular(
        state[:, 0, ...],
        stride,
        config.mesh_min_frac,
        config.mesh_max_frac,
        dilation_iters=0,
    )

    if np.any(masked):
      state = np.zeros((2, 1, box.size[1], box.size[0]))
      state = self.maybe_update_init_state(state, prev, config.options)

    return state

  def relax_mesh(
      self,
      x: np.ndarray,
      prev: np.ndarray,
      integration_config: mesh_lib.IntegrationConfig,
      mask: np.ndarray | None,
  ) -> tuple[np.ndarray, list[float], int, SolutionStatus]:
    """Performs mesh relaxation.

    This moves the mesh nodes to balance internal forces with forces pulling
    it into alignment with other sections, as established by the optical flow
    field.

    Args:
      x: [2, 1, y, x] initial XY coordinates of the mesh nodes to optimize
      prev: [2, 1, y, x] fixed XY coordinates of reference section(s)
      integration_config: configuration options for mesh relaxation
      mask: [1, y, x] optional boolean mask; mesh nodes corresponding to True
        entries will be set to nan

    Returns:
      tuple of:
        [2, 1, y, x] array of optimized mesh node positions
        list of kinetic energy history recorded during simulation
        number of steps simulated
        final solution status
    """
    config = self._config

    if mask is not None:
      flow_utils.apply_mask(x, mask)

    logging.info('Starting mesh relaxation with: %r', config)

    x, e_kin, num_steps = mesh_lib.relax_mesh(x, prev, integration_config)  # pytype: disable=wrong-arg-types  # jax-ndarray
    x = np.array(x)
    orig_x = x.copy()

    masked = map_utils.mask_irregular(
        x[:, 0, ...],
        integration_config.stride,
        config.mesh_min_frac,
        dilation_iters=5,
    )
    if not np.any(masked):
      return x, e_kin, num_steps, SolutionStatus.REGULAR

    logging.info('Attempting relaxation with 10% k0.')

    # Generate an a new initial state that is simiilar to the previous solution
    # everywhere except in the vicinity of the irregular nodes. If this
    # successfully generates a regular mesh, try simulation again. Otherwise
    # return the original solution.
    start_x = np.zeros_like(x)
    start_x = self.maybe_update_init_state(start_x, prev, config.options)

    x, _, prep_steps = mesh_lib.relax_mesh(  # pytype: disable=wrong-arg-types  # jax-ndarray
        start_x,
        x,
        dataclasses.replace(
            integration_config, k0=integration_config.k0 / 10.0
        ),
    )
    x = np.array(x)
    masked = map_utils.mask_irregular(
        x[:, 0, ...], integration_config.stride, config.mesh_min_frac
    )
    if np.any(masked):
      return (
          orig_x,
          e_kin,
          num_steps + prep_steps,
          SolutionStatus.PREP_FAILED,
      )

    if mask is not None:
      flow_utils.apply_mask(x, mask)

    x, e_kin2, reg_steps = mesh_lib.relax_mesh(x, prev, integration_config)  # pytype: disable=wrong-arg-types  # jax-ndarray
    x = np.array(x)
    return (
        x,
        e_kin2,
        num_steps + prep_steps + reg_steps,
        SolutionStatus.REGULARIZED,
    )

  def run_relaxation(
      self,
      bbox: bounding_box.BoundingBox,
  ) -> tuple[np.ndarray, list[float], int, SolutionStatus]:
    """Performs mesh relaxation."""
    config = self._config
    z = bbox.start[2]
    e_kin = []
    num_steps = 0
    status = SolutionStatus.UNDEFINED
    integration_config = config.integration_config
    # counters = Counters()
    prev = None
    mask = None

    # The first section in the block does not get optimized.
    if z not in config.block_starts:
      # with timer_counter(counters, 'load-mask'):
      if config.mask is not None:
        mask = self._build_mask(config.mask, bbox)

      # with timer_counter(counters, 'load-prev'):
      prev = self.get_prev_state(integration_config.stride, bbox)

    # with timer_counter(counters, 'load-mesh-init'):
    x = self.get_mesh_state(bbox, integration_config.stride, prev)

    if (
        z not in config.block_starts
        and not np.all(np.isnan(x))
        and prev is not None
        and not np.all(np.isnan(prev))
    ):
      # with timer_counter(counters, 'relax-mesh'):
      x, e_kin, num_steps, status = self.relax_mesh(
          x, prev, integration_config, mask
      )
    return x, e_kin, num_steps, status

  def process(self, subvol: Subvolume) -> Subvolume:
    bbox = subvol.bbox
    x, *_ = self.run_relaxation(bbox)
    return Subvolume(x, bbox)
