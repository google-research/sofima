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
from typing import Any

from connectomics.common import utils
import dataclasses_json
from sofima.processor import flow
from sofima.processor.defaults import em_2d


@dataclasses.dataclass(frozen=True)
class FlowPipelineConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for end-to-end SOFIMA flow pipelines."""

  estimate_flow: flow.EstimateFlow.Config
  reconcile_flows: flow.ReconcileAndFilterFlows.Config
  estimate_missing_flow: flow.EstimateMissingFlow.Config
  reconcile_missing_flows: flow.ReconcileAndFilterFlows.Config


def default_em_2d(
    overrides: dict[str, Any] | None = None,
) -> FlowPipelineConfig:
  """Default flow pipeline configuration for EM 2D data."""

  reconcile_missing_flows = utils.update_dataclass(
      em_2d.reconcile_flows_config(),
      {
          'multi_section': True,
          'max_magnitude': 0,
          'max_deviation': 10,
          'max_gradient': 10,
          'min_patch_size': 400,
          'base_delta_z': 1,
      },
  )

  config = FlowPipelineConfig(
      estimate_flow=em_2d.estimate_flow_config(),
      reconcile_flows=em_2d.reconcile_flows_config(),
      estimate_missing_flow=em_2d.estimate_missing_flow_config(),
      reconcile_missing_flows=reconcile_missing_flows,
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config
