# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

from absl.testing import absltest
from connectomics.common import utils
from connectomics.volume import subvolume_processor
from sofima.pipeline import flow_config
from sofima.processor.defaults import em_2d


def _expected_default_em_2d(overrides=None) -> flow_config.FlowPipeline:
  default_config = flow_config.FlowPipeline.from_dict({
      'estimate_flow': {
          'config': em_2d.estimate_flow_config(),
          'processing': {
              'overlap': [160, 160, 1],
              'subvolume_size': [3200, 3200, 128],
          },
          'schedule_batch_size': 16384,
          'corner_whitelist': set(),
          'ignore_existing': False,
          'delete_existing': False,
      },
      'reconcile_flows': em_2d.reconcile_flows_config(),
      'estimate_missing_flow': em_2d.estimate_missing_flow_config(),
      'reconcile_missing_flows': em_2d.reconcile_flows_config({
          'multi_section': True,
          'max_magnitude': 0,
          'max_deviation': 10,
          'max_gradient': 10,
          'min_patch_size': 400,
          'base_delta_z': 1,
      }),
  })
  if overrides is not None:
    default_config = utils.update_dataclass(default_config, overrides)
  return default_config


class FlowConfigTest(absltest.TestCase):

  def test_default_em_2d(self):
    config = subvolume_processor.default_config(
        flow_config.FlowPipeline,
        subvolume_processor.DefaultConfigType.EM_2D,
    )
    self.assertEqual(
        config,
        _expected_default_em_2d(),
    )

    config = subvolume_processor.default_config(
        flow_config.FlowPipeline,
        subvolume_processor.DefaultConfigType.EM_2D,
        {
            'estimate_flow': {
                'config': {'z_stride': 12321},
                'processing': {
                    'subvolume_size': [1000, 1000, 1],
                },
            }
        },
    )
    self.assertEqual(
        config,
        _expected_default_em_2d(
            {
                'estimate_flow': {
                    'config': {'z_stride': 12321},
                    'processing': {
                        'overlap': [160, 160, 12321],
                        'subvolume_size': [1000, 1000, 1],
                    },
                },
            },
        ),
    )


if __name__ == '__main__':
  absltest.main()
