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

"""Tests for flow_utils."""

from absl.testing import absltest
import numpy as np
from sofima import flow_utils


class FlowUtilsTest(absltest.TestCase):

  def test_apply_mask(self):
    flow = np.zeros((3, 1, 50, 50))
    mask = np.zeros((1, 50, 50), dtype=bool)
    mask[0, 10, 15] = True
    mask[0, 3, 4] = True
    flow_utils.apply_mask(flow, mask)

    expected = np.zeros((3, 1, 50, 50))
    expected[:, 0, 10, 15] = np.nan
    expected[:, 0, 3, 4] = np.nan

    np.testing.assert_array_equal(flow, expected)

  def test_clean_flow(self):
    flow = np.zeros((4, 1, 50, 40))
    flow[2, ...] = 2.0
    flow[2, 0, 10, 20] = 1.2
    flow[3, 0, 10, 22] = 1.2
    flow[3, 0, 10, 24] = 1.6
    flow[0, 0, 5, 4] = 12
    flow[1, 0, 5, 6] = -14
    flow[:, 0, 5, 10] = 2
    flow[:, 0, 15, 10] = 7

    cleaned = flow_utils.clean_flow(
        flow,
        min_peak_ratio=1.4,
        min_peak_sharpness=1.6,
        max_magnitude=10,
        max_deviation=5)

    expected = np.zeros((2, 1, 50, 40))
    expected[:, 0, 5, 10] = 2
    expected[:, 0, 15, 10] = np.nan  # median filter
    expected[:, 0, 10, 20] = np.nan  # peak sharpness
    expected[:, 0, 10, 22] = np.nan  # peak ratio
    expected[:, 0, 5, 4] = np.nan  # magnitude
    expected[:, 0, 5, 6] = np.nan  # magnitude

    np.testing.assert_array_equal(cleaned, expected)

  def test_reconcile_flows(self):
    flow1 = np.full((3, 1, 50, 40), np.nan)
    flow2 = np.full((3, 1, 50, 40), np.nan)
    flow3 = np.full((3, 1, 50, 40), np.nan)

    flow1[:, 0, 10, 10] = 2.
    flow2[:, 0, 10, 10] = 3.  # ignored, flow1 preferred.

    flow3[:, 0, 20, 20] = 4.
    flow2[:, 0, 20, 20] = 1.  # ignored, min_delta_z.

    flow2[:, 0, 30:35, 30:35] = 5
    flow2[0, 0, 32, 32] = 15  # ignored, max_deviation.

    reconciled = flow_utils.reconcile_flows([flow1, flow2, flow3],
                                            max_gradient=0,
                                            max_deviation=8,
                                            min_patch_size=0,
                                            min_delta_z=2)

    expected = np.full((3, 1, 50, 40), np.nan)
    expected[:, 0, 10, 10] = 2.
    expected[:, 0, 20, 20] = 4.
    expected[:, 0, 30:35, 30:35] = 5
    expected[:, 0, 32, 32] = np.nan

    np.testing.assert_array_equal(reconciled, expected)


if __name__ == '__main__':
  absltest.main()
