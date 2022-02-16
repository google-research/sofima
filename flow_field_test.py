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
"""Tests for flow_field."""

from absl.testing import absltest
import numpy as np
from sofima import flow_field


class FlowFieldTest(absltest.TestCase):

  def test_jax_masked_xcorr_calculator(self):
    pre_image = np.zeros((120, 120), dtype=np.uint8)
    post_image = np.zeros((120, 120), dtype=np.uint8)

    pre_image[60, 60] = 255
    post_image[70, 53] = 255

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    field = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4)

    np.testing.assert_array_equal([4, 2, 2], field.shape)
    np.testing.assert_array_equal(7 * np.ones((2, 2)), field[0, ...])
    np.testing.assert_array_equal(-10 * np.ones((2, 2)), field[1, ...])
    np.testing.assert_array_equal(np.zeros((2, 2)), field[3, ...])

    # 2nd point in the post-image would normally confuse the flow estimation,
    # but with masking it should have no impact.
    post_image[54, 68] = 255
    post_image_mask = np.zeros((128, 128), dtype=bool)
    post_image_mask[:55, :70] = 1
    field = calculator.flow_field(
        pre_image,
        post_image,
        patch_size=80,
        step=40,
        post_mask=post_image_mask,
        batch_size=4)

    np.testing.assert_array_equal([4, 2, 2], field.shape)
    np.testing.assert_array_equal(7 * np.ones((2, 2)), field[0, ...])
    np.testing.assert_array_equal(-10 * np.ones((2, 2)), field[1, ...])
    np.testing.assert_array_equal(np.zeros((2, 2)), field[3, ...])

  def test_jax_xcorr_3d(self):
    pre_image = np.zeros((50, 100, 100), dtype=np.uint8)
    post_image = np.zeros((50, 100, 100), dtype=np.uint8)

    pre_image[25, 50, 50] = 255
    post_image[22, 45, 54] = 255

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flow = calculator.flow_field(
        pre_image, post_image, patch_size=(40, 80, 80), step=10, batch_size=1)

    np.testing.assert_array_equal([5, 2, 3, 3], flow.shape)
    np.testing.assert_array_equal(np.full([2, 3, 3], -4), flow[0, ...])
    np.testing.assert_array_equal(np.full([2, 3, 3], 5), flow[1, ...])
    np.testing.assert_array_equal(np.full([2, 3, 3], 3), flow[2, ...])

  def test_jax_peak(self):
    hy, hx = np.mgrid[:50, :50]
    cy, cx = 20, 28
    hy = cy - hy
    hx = cx - hx
    r = np.sqrt(2 * hx**2 + hy**2)
    peak_max = 10
    xcorr = peak_max * np.exp(-r / 4)

    peaks = flow_field._batched_peaks(
        xcorr[np.newaxis, ...], (25, 25),
        min_distance=2,
        threshold_rel=0.5,
        peak_radius=(2, 3))
    np.testing.assert_array_equal([1, 4], peaks.shape)

    peak_support = np.min(xcorr[cy - 2:cy + 3, cx - 3:cx + 4])
    self.assertEqual(peaks[0, 0], 3)  # x
    self.assertEqual(peaks[0, 1], -5)  # y
    self.assertEqual(peaks[0, 2], peak_max / peak_support)  # sharpness
    self.assertEqual(peaks[0, 3], 0)  # peak ratio


if __name__ == '__main__':
  absltest.main()
