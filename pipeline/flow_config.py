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
"""Configuration for SOFIMA flow pipelines.

As implemented, these are reasonable defaults for aligning 2d sections e.g. EM
data. For 3d data, the default values will likely need to be adjusted.
"""

import dataclasses

import dataclasses_json
from sofima.processor import flow


EstimateFlowConfig = flow.EstimateFlow.EstimateFlowConfig
ReconcileFlowsConfig = flow.ReconcileAndFilterFlows.ReconcileFlowsConfig
EstimateMissingFlowConfig = flow.EstimateMissingFlow.EstimateMissingFlowConfig


def _default_estimate_flow_config() -> EstimateFlowConfig:
  return EstimateFlowConfig(patch_size=160, stride=40, z_stride=1)


def _default_reconcile_flows_config() -> ReconcileFlowsConfig:
  return ReconcileFlowsConfig(
      min_peak_sharpness=1.4,
      min_peak_ratio=1.4,
      max_magnitude=80,
      max_deviation=5,
      max_gradient=5,
      min_patch_size=400,
      multi_section=False,
      base_delta_z=0,
  )


def _default_estimate_missing_flow_config() -> EstimateMissingFlowConfig:
  return EstimateMissingFlowConfig(
      patch_size=160,
      stride=40,
      delta_z=1,
      max_delta_z=4,
      max_magnitude=80,
      image_volinfo=None,
  )


@dataclasses.dataclass(frozen=True)
class FlowPipelineConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for end-to-end SOFIMA flow pipelines."""

  estimate_flow: EstimateFlowConfig = dataclasses.field(
      default_factory=_default_estimate_flow_config,
  )
  reconcile_flows: ReconcileFlowsConfig | None = dataclasses.field(
      default_factory=_default_reconcile_flows_config,
  )
  estimate_missing_flow: EstimateMissingFlowConfig | None = dataclasses.field(
      default_factory=_default_estimate_missing_flow_config,
  )
  reconcile_missing_flows: ReconcileFlowsConfig | None = dataclasses.field(
      default_factory=_default_reconcile_flows_config,
  )
