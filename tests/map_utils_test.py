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

"""Tests for map_utils."""

from absl.testing import absltest
from connectomics.common import bounding_box
import numpy as np
from scipy import interpolate
from sofima import map_utils


class MapUtilsTest(absltest.TestCase):

  def test_interpolate_points(self):
    coord_map = 2.5 * np.random.random((2, 1, 10, 10))
    coord_map[:, 0, 4, 3] = np.nan
    coord_map[:, 0, 2, 6] = np.nan
    hy, hx = np.mgrid[:coord_map.shape[2], :coord_map.shape[3]]
    query_points = (hx.ravel() + np.random.random(hx.size),
                    hy.ravel() + np.random.random(hy.size))
    valid = np.all(np.isfinite(coord_map[:, 0, ...]), axis=0)
    data_points = hx[valid], hy[valid]

    expected_u = interpolate.griddata(
        data_points, coord_map[0, 0, ...][valid], query_points, method='linear')
    expected_v = interpolate.griddata(
        data_points, coord_map[1, 0, ...][valid], query_points, method='linear')

    u, v = map_utils._interpolate_points(data_points, query_points,
                                         coord_map[0, 0, ...][valid],
                                         coord_map[1, 0, ...][valid])

    np.testing.assert_array_equal(u, expected_u)
    np.testing.assert_array_equal(v, expected_v)

  def test_abs_rel_conversion(self):
    np.random.seed(11111)
    rel_coord = np.random.uniform(-0.5, 0.5, [2, 1, 50, 50])
    stride = 10

    # relative -> absolute -> relative conversion.
    abs_coord = map_utils.to_absolute(rel_coord, stride)
    np.testing.assert_allclose(
        map_utils.to_relative(abs_coord, stride), rel_coord)

    # Same as above, but with a custom origin point.
    box = bounding_box.BoundingBox(start=(240, 280, 300), size=(50, 50, 1))
    abs_coord = map_utils.to_absolute(rel_coord, stride, box)
    np.testing.assert_allclose(
        map_utils.to_relative(abs_coord, stride, box), rel_coord)

  def test_fill_missing(self):
    hy, hx = np.mgrid[:50, :50]
    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[0, 0, ...] = np.sin(hx / 25)
    coord_map[1, 0, ...] = np.cos(hy / 25)

    with_gap = coord_map.copy()
    with_gap[:, 0, 24:28, 38:42] = np.nan

    filled = map_utils.fill_missing(with_gap)
    np.testing.assert_array_almost_equal(filled, coord_map, decimal=2)

    with_gap = coord_map.copy()
    with_gap[:, 0, -1, :] = np.nan
    filled = map_utils.fill_missing(with_gap)
    self.assertTrue(np.all(np.isnan(filled[:, 0, -1, :])))

    filled = map_utils.fill_missing(with_gap, extrapolate=True)
    np.testing.assert_array_almost_equal(
        filled[1, 0, -1, :], coord_map[1, 0, -1, :], decimal=1)

    with_gap[...] = np.nan
    filled = map_utils.fill_missing(with_gap, invalid_to_zero=True)
    self.assertTrue(np.all(filled == 0))

  def test_outer_box(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[0, 0, 0, 49] = 4
    coord_map[0, 0, 1, 49] = 8
    coord_map[0, 0, 2, 0] = -3
    coord_map[1, 0, 49, 10] = 1
    coord_map[1, 0, 0, 1] = -2
    outer_box = map_utils.outer_box(coord_map, box, stride=5)

    self.assertEqual(
        outer_box,
        bounding_box.BoundingBox(start=(99, 199, 10), size=(53, 52, 1)))

  def test_inner_box(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[1, :, ...] = -30
    coord_map[1, :, 0, :] = -40
    coord_map[1, :, -1, :] = -25
    inner_box = map_utils.inner_box(coord_map, box, stride=10)

    self.assertEqual(
        inner_box,
        bounding_box.BoundingBox(start=(100, 196, 10), size=(50, 51, 1)))

    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[0, :, :, 0] = -9
    coord_map[0, :, :, -1] = 9
    inner_box = map_utils.inner_box(coord_map, box, stride=10)
    self.assertEqual(
        inner_box,
        bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1)))

  def test_invert_map(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    _, hx = np.mgrid[:50, :50]
    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[1, 0, ...] = np.sin(hx / 25) * 20

    inv_map = map_utils.invert_map(coord_map, box, box, 40.)

    np.testing.assert_array_almost_equal(
        inv_map[:, :, 1:, 1:], -coord_map[:, :, 1:, 1:], decimal=5)

  def test_resample_map(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    hy, hx = np.mgrid[:50, :50]
    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[0, 0, ...] = np.sin(hx / 25) * 20
    coord_map[1, 0, ...] = np.cos(hy / 25) * 20

    hy, hx = np.mgrid[:100, :100]
    expected = np.zeros([2, 1, 100, 100])
    expected[0, 0, ...] = np.sin(hx / 50) * 20
    expected[1, 0, ...] = np.cos(hy / 50) * 20

    dst_box = bounding_box.BoundingBox(start=(102, 203, 10), size=(48, 47, 1))
    dst_box = dst_box.scale([2, 2, 1.0])  # adjust for output stride
    resampled = map_utils.resample_map(coord_map, box, dst_box, 40, 20)
    np.testing.assert_array_almost_equal(
        resampled[:, :, :-1, :-1], expected[:, :, 6:-1, 4:-1], decimal=2)

  def test_compose_maps(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    coord_map = np.zeros([2, 1, 50, 50])
    hy, hx = np.mgrid[:50, :50]
    coord_map[0, 0, ...] = np.sin(hx / 25)
    coord_map[1, 0, ...] = np.cos(hy / 25)
    stride = 5

    # Composing a map with its inversion should yield the identity transform.
    inverted = map_utils.invert_map(coord_map, box, box, stride)
    composed = map_utils.compose_maps(coord_map, box, stride, inverted, box,
                                      stride)[:, :, 1:-2, 1:-2]

    np.testing.assert_array_almost_equal(
        composed, np.zeros_like(composed), decimal=3)

  def test_compose_maps_fast(self):
    coord_map = np.zeros([2, 1, 60, 60])
    flow = np.zeros([2, 1, 50, 50])
    flow[0, 0, :, 10:25] = -5
    flow[0, 0, :, 25:40] = 65
    flow[:, 0, :, 4] = np.nan
    stride = 40

    box1 = bounding_box.BoundingBox(start=(42, 58, 64), size=(50, 50, 1))
    box2 = bounding_box.BoundingBox(start=(40, 50, 64), size=(60, 60, 1))

    # coord_map is identity, so nothing should change in the flow.
    updated = np.array(
        map_utils.compose_maps_fast(flow, box1.start[::-1], stride, coord_map,
                                    box2.start[::-1], stride))
    np.testing.assert_array_equal(updated, flow)

    # Now test with a non-zero coordinate map.
    coord_map[0, :, :, 7:] = -10
    updated = np.array(
        map_utils.compose_maps_fast(flow, box1.start[::-1], stride, coord_map,
                                    box2.start[::-1], stride))
    flow[0, 0, :, 5:10] = -10
    flow[0, 0, :, 10:25] = -15
    flow[0, 0, :, 25:40] = 55
    flow[0, 0, :, 40:] = -10
    np.testing.assert_array_equal(updated, flow)

    # Also test on the same case as the slow version of the function.
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    coord_map = np.zeros([2, 1, 50, 50])
    hy, hx = np.mgrid[:50, :50]
    coord_map[0, 0, ...] = np.sin(hx / 25)
    coord_map[1, 0, ...] = np.cos(hy / 25)
    stride = 5

    # Composing a map with its inversion should yield the identity transform.
    inverted = map_utils.invert_map(coord_map, box, box, stride)
    composed = np.array(
        map_utils.compose_maps_fast(coord_map, box.start[::-1], stride,
                                    inverted, box.start[::-1],
                                    stride))[:, :, 1:-2, 1:-2]

    np.testing.assert_array_almost_equal(
        composed, np.zeros_like(composed), decimal=3)

  def test_mask_irregular(self):
    coord_map = np.zeros([2, 50, 50])
    coord_map[0, 40, 10] = 10
    bad = map_utils.mask_irregular(coord_map, 40, 0.25, 1.1)

    expected = np.zeros([2, 50, 50])
    expected[:, 39:42, 8:11] = np.nan

    np.testing.assert_array_equal(expected, coord_map)
    np.testing.assert_array_equal(np.isnan(expected[0, ...]), bad)


if __name__ == '__main__':
  absltest.main()
