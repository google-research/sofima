# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Default configurations for 2D EM processors."""

from typing import Any

from connectomics.common import utils
from connectomics.volume import subvolume_processor
from sofima import mesh as mesh_lib
from sofima.processor import flow
from sofima.processor import maps
from sofima.processor import mesh
from sofima.processor import warp


def estimate_flow_config(
    overrides: dict[str, Any] | None = None,
) -> flow.EstimateFlow.Config:
  """Default configuration for estimating flow fields in EM 2D data."""
  config = flow.EstimateFlow.Config(
      patch_size=160,
      stride=40,
      z_stride=1,
      fixed_current=False,
      mask_configs=None,
      mask_only_for_patch_selection=True,
      selection_mask_configs=None,
      batch_size=1024,
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


def reconcile_flows_config(
    overrides: dict[str, Any] | None = None,
) -> flow.ReconcileAndFilterFlows.Config:
  """Default configuration for reconciling flow fields in EM 2D data."""
  config = flow.ReconcileAndFilterFlows.Config(
      flow_volinfos=None,
      mask_configs=None,
      min_peak_ratio=1.6,
      min_peak_sharpness=1.6,
      max_magnitude=40,
      max_deviation=10,
      max_gradient=40,
      min_patch_size=400,
      multi_section=False,
      base_delta_z=1,
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


def estimate_missing_flow_config(
    overrides: dict[str, Any] | None = None,
) -> flow.EstimateMissingFlow.Config:
  """Default configuration for estimating missing flow fields in EM 2D data."""
  config = flow.EstimateMissingFlow.Config(
      patch_size=160,
      stride=40,
      delta_z=1,
      max_delta_z=4,
      max_attempts=2,
      mask_configs=None,
      mask_only_for_patch_selection=True,
      selection_mask_configs=None,
      min_peak_ratio=1.6,
      min_peak_sharpness=1.6,
      max_magnitude=40,
      batch_size=1024,
      image_volinfo=None,
      image_cache_bytes=int(1e9),
      mask_cache_bytes=int(1e9),
      search_radius=0,
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


subvolume_processor.register_default_config(
    subvolume_processor.DefaultConfigType.EM_2D,
    flow.EstimateFlow.Config,
    estimate_flow_config,
)
subvolume_processor.register_default_config(
    subvolume_processor.DefaultConfigType.EM_2D,
    flow.ReconcileAndFilterFlows.Config,
    reconcile_flows_config,
)
subvolume_processor.register_default_config(
    subvolume_processor.DefaultConfigType.EM_2D,
    flow.EstimateMissingFlow.Config,
    estimate_missing_flow_config,
)


def relax_mesh_config(
    overrides: dict[str, Any] | None = None,
) -> mesh.RelaxMesh.Config:
  """Default mesh relaxation configuration for EM 2D data."""

  config = mesh.RelaxMesh.Config(
      output_dir='NONE',
      integration_config=mesh_lib.IntegrationConfig(
          dt=0.001,
          gamma=0.0,
          k0=0.01,
          k=0.1,
          stride=(40, 40),
          num_iters=1000,
          max_iters=100000,
          stop_v_max=0.005,
          dt_max=1000,
          start_cap=0.01,
          final_cap=10,
          prefer_orig_order=True,
      ),
      mesh=None,
      flows=[],
      sections_to_skip=[],
      ranges_to_skip=[],
      mask=None,
      block_starts=[],
      block_ends=[],
      backward=False,
      mesh_min_frac=0.5,
      mesh_max_frac=2.0,
      coming_in=[],
      options=mesh.MeshOptions(
          irregular_mask_radius=5,
      ),
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


subvolume_processor.register_default_config(
    subvolume_processor.DefaultConfigType.EM_2D,
    flow.EstimateFlow.Config,
    estimate_flow_config,
)


def within_block_config(
    overrides: dict[str, Any] | None = None,
) -> mesh.RelaxMesh.Config:
  """Default configuration for within-block mesh relaxation."""
  config = relax_mesh_config()
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


def last_section_config(
    overrides: dict[str, Any] | None = None,
) -> mesh.RelaxMesh.Config:
  """Default configuration for relaxation of last section of blockwise mesh."""
  config = relax_mesh_config()
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


def cross_block_config(
    overrides: dict[str, Any] | None = None,
) -> mesh.RelaxMesh.Config:
  """Default cross-block mesh relaxation configuration for EM 2D data."""
  config = relax_mesh_config({
      'integration_config': {
          'k0': 0.001,
          'stride': (320, 320),
          'stop_v_max': 0.001,
      },
      'options': {
          'init_state': mesh.MeshInitState.PREV_MEDIAN,
      },
  })
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


def default_em_2d_reconcile_config(
    overrides: dict[str, Any] | None = None,
) -> maps.ReconcileCrossBlockMaps.Config:
  """Default cross-block reconciliation configuration for EM 2D data."""
  config = maps.ReconcileCrossBlockMaps.Config(
      cross_block='NONE',
      cross_block_inv='NONE',
      last_inv='NONE',
      main_inv='NONE',
      z_map={},
      stride=40,
      xy_overlap=128,
      backward=False,
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


def warp_config(
    overrides: dict[str, Any] | None = None,
) -> warp.WarpByMap.Config:
  """Default warp configuration for EM 2D data."""
  config = warp.WarpByMap.Config(
      stride=40,
      map_volinfo='UNSET',
      data_volinfo='UNSET',
      map_decorator_specs=None,
      data_decorator_specs=None,
      map_scale=1.0,
      interpolation='nearest',
      downsample=1,
      offset=0.0,
      mask_configs=None,
      source_cache_bytes=int(1e9),
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config
