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


@dataclasses.dataclass(frozen=True)
class FlowPipelineConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for end-to-end SOFIMA flow pipelines."""

  estimate_flow: EstimateFlowConfig = dataclasses.field(
      default_factory=EstimateFlowConfig,
  )
  reconcile_flows: ReconcileFlowsConfig = dataclasses.field(
      default_factory=ReconcileFlowsConfig,
  )
  estimate_missing_flow: EstimateMissingFlowConfig = dataclasses.field(
      default_factory=EstimateMissingFlowConfig,
  )
  reconcile_missing_flows: ReconcileFlowsConfig = dataclasses.field(
      default_factory=ReconcileFlowsConfig,
  )
