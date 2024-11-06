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
import enum
from typing import Any

from connectomics.common import utils
import dataclasses_json
from sofima import mesh as mesh_lib
from sofima.processor import maps
from sofima.processor import mesh


# TODO(blakely): Combine with flow_config.DefaultPipeline
class DefaultPipeline(enum.Enum):
  EM_2D = 'em_2d'


@dataclasses.dataclass(frozen=True)
class MeshRelaxationConfig(dataclasses_json.DataClassJsonMixin):
  """Pipeline configuration for mesh relaxation."""

  within_block_config: mesh.RelaxMesh.Config
  last_section_config: mesh.RelaxMesh.Config
  cross_block_config: mesh.RelaxMesh.Config
  reconcile_cross_block_config: maps.ReconcileCrossBlockMaps.Config


def default_em_2d_relax_mesh_config(
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


def default_em_2d_within_block_config(
    overrides: dict[str, Any] | None = None,
) -> mesh.RelaxMesh.Config:
  """Default configuration for within-block mesh relaxation."""
  config = default_em_2d_relax_mesh_config()
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


def default_em_2d_last_section_config(
    overrides: dict[str, Any] | None = None,
) -> mesh.RelaxMesh.Config:
  """Default configuration for relaxation of last section of blockwise mesh."""
  config = default_em_2d_relax_mesh_config()
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


def default_em_2d_cross_block_config(
    overrides: dict[str, Any] | None = None,
) -> mesh.RelaxMesh.Config:
  """Default cross-block mesh relaxation configuration for EM 2D data."""
  config = default_em_2d_relax_mesh_config({
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


def default_em_2d(
    overrides: dict[str, Any] | None = None,
) -> MeshRelaxationConfig:
  """Default mesh relaxation configuration for EM 2D data."""
  config = MeshRelaxationConfig(
      within_block_config=default_em_2d_within_block_config(),
      last_section_config=default_em_2d_last_section_config(),
      cross_block_config=default_em_2d_cross_block_config(),
      reconcile_cross_block_config=default_em_2d_reconcile_config(),
  )
  if overrides is not None:
    config = utils.update_dataclass(config, overrides)
  return config


_DEFAULT_CONFIG_TYPE_DISPATCH = {
    DefaultPipeline.EM_2D: default_em_2d,
}


def default(
    default_type: DefaultPipeline, overrides: dict[str, Any] | None = None
) -> MeshRelaxationConfig:
  """Default mesh relaxation configuration for a given data type."""
  return _DEFAULT_CONFIG_TYPE_DISPATCH[default_type](overrides)
