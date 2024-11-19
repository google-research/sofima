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
from sofima.processor import flow


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
