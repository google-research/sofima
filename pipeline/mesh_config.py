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
"""Configuration for SOFIMA mesh relaxation pipelines.

As implemented, these are reasonable defaults for relaxing flows based on 2d
sections e.g. EM data. For 3d data, the default values will likely need to be
adjusted.
"""

import dataclasses
from typing import Any

from connectomics.common import utils
from connectomics.volume import subvolume_processor
import dataclasses_json
from sofima.processor import maps
from sofima.processor import mesh
from sofima.processor.defaults import em_2d


@dataclasses.dataclass(frozen=True)
class MeshRelaxationConfig(dataclasses_json.DataClassJsonMixin):
  """Pipeline configuration for mesh relaxation."""

  within_block_config: mesh.RelaxMesh.Config
  last_section_config: mesh.RelaxMesh.Config
  cross_block_config: mesh.RelaxMesh.Config
  reconcile_cross_block_config: maps.ReconcileCrossBlockMaps.Config


def default_em_2d(
    overrides: dict[str, Any] | None = None,
) -> MeshRelaxationConfig:
  """Default mesh relaxation configuration for EM 2D data."""
  within_block = em_2d.within_block_config()
  last_section = em_2d.last_section_config()
  cross_block = em_2d.cross_block_config()
  reconcile_cross_block = em_2d.default_em_2d_reconcile_config()
  config = MeshRelaxationConfig(
      within_block_config=within_block,
      last_section_config=last_section,
      cross_block_config=cross_block,
      reconcile_cross_block_config=reconcile_cross_block,
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config

subvolume_processor.register_default_config(
    subvolume_processor.DefaultConfigType.EM_2D,
    MeshRelaxationConfig,
    default_em_2d,
)
