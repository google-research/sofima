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

"""Gradient-filter regressions for finite flow-field boundaries."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from sofima import flow_utils


def reconcile(flows, threshold=1.0, min_patch_size=0):
  return flow_utils.reconcile_flows(
      flows,
      max_gradient=threshold,
      max_deviation=0,
      min_patch_size=min_patch_size,
  )


class FlowBoundaryTest(parameterized.TestCase):

  @parameterized.parameters(
      {'shape': (2, 1, 4, 5)},
      {'shape': (3, 2, 4, 5)},
      {'shape': (2, 1, 1, 5)},
      {'shape': (2, 1, 4, 1)},
      {'shape': (3, 1, 1, 1)},
  )
  def test_constant_translations_are_preserved_at_all_boundaries(self, shape):
    flow = np.empty(shape, dtype=np.float32)
    for channel, value in enumerate((50.0, -70.0, 3.0)[: shape[0]]):
      flow[channel] = value
    before = flow.copy()
    np.testing.assert_array_equal(reconcile([flow]), flow)
    np.testing.assert_array_equal(flow, before)

  @parameterized.parameters(0, 1)
  def test_step_filters_both_sides_without_inventing_edge_discontinuities(
      self, channel
  ):
    flow = np.zeros((2, 1, 4, 5))
    if channel == 0:
      flow[0, ..., 2:] = 3.0
      bad = np.zeros((1, 4, 5), bool)
      bad[..., 1:3] = True
    else:
      flow[1, :, 2:, :] = 3.0
      bad = np.zeros((1, 4, 5), bool)
      bad[:, 1:3, :] = True
    actual = reconcile([flow])
    np.testing.assert_array_equal(
        np.isnan(actual), np.broadcast_to(bad, flow.shape)
    )
    np.testing.assert_array_equal(actual[:, ~bad], flow[:, ~bad])

  def test_gradient_filter_is_invariant_to_constant_offsets(self):
    flow = np.zeros((3, 2, 5, 6))
    flow[0, :, :, 3:] = 4.0
    flow[1, :, 3:, :] = -3.0
    offset = np.array([80.0, -90.0, 2.0])[:, None, None, None]
    unshifted = reconcile([flow])
    shifted = reconcile([flow + offset])
    np.testing.assert_array_equal(np.isnan(shifted), np.isnan(unshifted))
    np.testing.assert_allclose(shifted - offset, unshifted, equal_nan=True)

  def test_fallback_filling_and_patch_filter_retain_complete_translation(self):
    flow = np.full((2, 1, 4, 5), 20.0)
    primary = flow.copy()
    primary[:, :, 1:3, 2:4] = np.nan
    actual = reconcile([primary, flow], min_patch_size=20)
    np.testing.assert_array_equal(actual, flow)
    self.assertTrue(np.isnan(primary[:, :, 1:3, 2:4]).all())

  def test_existing_missing_values_and_disabled_filter_are_preserved(self):
    flow = np.full((2, 1, 3, 4), 50.0)
    flow[:, :, 1, 2] = np.nan
    np.testing.assert_array_equal(reconcile([flow], threshold=0), flow)
    np.testing.assert_array_equal(reconcile([flow]), flow)

  def test_exact_threshold_does_not_remove_neighboring_points(self):
    flow = np.zeros((2, 1, 3, 4))
    flow[0] = np.arange(4)[None, None, :] + 50.0
    flow[1] = np.arange(3)[None, :, None] - 50.0
    np.testing.assert_array_equal(reconcile([flow]), flow)


if __name__ == '__main__':
  absltest.main()
