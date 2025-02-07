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
"""Configuration for rendering/warping a volume."""

import dataclasses
from typing import Any

from connectomics.common import utils
from connectomics.volume import subvolume_processor
import dataclasses_json
from sofima.processor import warp
from sofima.processor.defaults import em_2d


@dataclasses.dataclass(frozen=True)
class WarpPipelineConfig(dataclasses_json.DataClassJsonMixin):
  """Configuration for warping a volume."""

  warp: warp.WarpByMap.Config


def default_em_2d(
    overrides: dict[str, Any] | None = None,
) -> WarpPipelineConfig:
  """Default warp configuration for EM 2D data."""
  config = WarpPipelineConfig(
      warp=em_2d.warp_config(),
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


subvolume_processor.register_default_config(
    subvolume_processor.DefaultConfigType.EM_2D,
    WarpPipelineConfig,
    default_em_2d,
)
