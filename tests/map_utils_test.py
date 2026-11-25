# coding=utf-8
# Copyright 2022-2023 The Google Research Authors.
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

from unittest import mock

from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.volume import subvolume
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy import spatial
from sofima import map_utils
from sofima.processor import maps as maps_proc


class _ReconcileWithMemVolumes(maps_proc.ReconcileCrossBlockMaps):
  """ReconcileCrossBlockMaps subclass backed by in-memory arrays."""

  def __init__(self, config, volumes):
    super().__init__(config)
    self._volumes = volumes

  def _open_volume(self, path):
    return self._volumes[path]


class MapUtilsTest(absltest.TestCase):

  def test_interpolate_points(self):
    coord_map = 2.5 * np.random.random((2, 1, 10, 10))
    coord_map[:, 0, 4, 3] = np.nan
    coord_map[:, 0, 2, 6] = np.nan
    hy, hx = np.mgrid[: coord_map.shape[2], : coord_map.shape[3]]
    query_points = (
        hx.ravel() + np.random.random(hx.size),
        hy.ravel() + np.random.random(hy.size),
    )
    valid = np.all(np.isfinite(coord_map[:, 0, ...]), axis=0)
    data_points = hx[valid], hy[valid]

    expected_u = interpolate.griddata(
        data_points, coord_map[0, 0, ...][valid], query_points, method='linear'
    )
    expected_v = interpolate.griddata(
        data_points, coord_map[1, 0, ...][valid], query_points, method='linear'
    )

    u, v = map_utils._interpolate_points(
        data_points,
        query_points,
        coord_map[0, 0, ...][valid],
        coord_map[1, 0, ...][valid],
    )

    np.testing.assert_array_equal(u, expected_u)
    np.testing.assert_array_equal(v, expected_v)

  def test_interpolate_points_3d(self):
    coord_map = 2.5 * np.random.random((3, 10, 10, 10))
    coord_map[:, 2:8, 4, 3] = np.nan
    coord_map[:, 2:8, 2, 6] = np.nan
    hz, hy, hx = np.mgrid[
        : coord_map.shape[1], : coord_map.shape[2], : coord_map.shape[3]
    ]
    query_points = (
        hx.ravel() + np.random.random(hx.size),
        hy.ravel() + np.random.random(hy.size),
        hz.ravel() + np.random.random(hz.size),
    )
    valid = np.all(np.isfinite(coord_map), axis=0)
    data_points = hx[valid], hy[valid], hz[valid]

    expected_u = interpolate.griddata(
        data_points, coord_map[0, ...][valid], query_points, method='linear'
    )
    expected_v = interpolate.griddata(
        data_points, coord_map[1, ...][valid], query_points, method='linear'
    )
    expected_w = interpolate.griddata(
        data_points, coord_map[2, ...][valid], query_points, method='linear'
    )

    u, v, w = map_utils._interpolate_points(
        data_points,
        query_points,
        coord_map[0, ...][valid],
        coord_map[1, ...][valid],
        coord_map[2, ...][valid],
    )

    np.testing.assert_array_equal(u, expected_u)
    np.testing.assert_array_equal(v, expected_v)
    np.testing.assert_array_equal(w, expected_w)

  def test_abs_rel_conversion(self):
    np.random.seed(11111)
    rel_coord = np.random.uniform(-0.5, 0.5, [2, 1, 50, 50])
    stride = 10

    # relative -> absolute -> relative conversion.
    abs_coord = map_utils.to_absolute(rel_coord, stride)
    np.testing.assert_allclose(
        map_utils.to_relative(abs_coord, stride), rel_coord
    )

    # Same as above, but with a custom origin point.
    box = bounding_box.BoundingBox(start=(240, 280, 300), size=(50, 50, 1))
    abs_coord = map_utils.to_absolute(rel_coord, stride, box)
    np.testing.assert_allclose(
        map_utils.to_relative(abs_coord, stride, box), rel_coord
    )

  def test_abs_rel_conversion_3d(self):
    np.random.seed(11111)
    rel_coord = np.random.uniform(-0.5, 0.5, [3, 25, 50, 50])
    stride = 7

    # relative -> absolute -> relative conversion.
    abs_coord = map_utils.to_absolute(rel_coord, stride)
    np.testing.assert_allclose(
        map_utils.to_relative(abs_coord, stride), rel_coord
    )

    # Same as above, but with a custom origin point.
    box = bounding_box.BoundingBox(start=(240, 280, 300), size=(50, 50, 25))
    abs_coord = map_utils.to_absolute(rel_coord, stride, box)
    np.testing.assert_allclose(
        map_utils.to_relative(abs_coord, stride, box), rel_coord
    )

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
        filled[1, 0, -1, :], coord_map[1, 0, -1, :], decimal=1
    )

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
        bounding_box.BoundingBox(start=(99, 199, 10), size=(53, 52, 1)),
    )

  def test_inner_box(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[1, :, ...] = -30
    coord_map[1, :, 0, :] = -40
    coord_map[1, :, -1, :] = -25
    inner_box = map_utils.inner_box(coord_map, box, stride=10)

    self.assertEqual(
        inner_box,
        bounding_box.BoundingBox(start=(100, 196, 10), size=(50, 51, 1)),
    )

    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[0, :, :, 0] = -9
    coord_map[0, :, :, -1] = 9
    inner_box = map_utils.inner_box(coord_map, box, stride=10)
    self.assertEqual(
        inner_box,
        bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1)),
    )

  def test_inner_box3d(self):
    box = bounding_box.BoundingBox(start=(100, 200, 200), size=(50, 50, 50))
    coord_map = np.zeros([3, 50, 50, 50])
    coord_map[2, ...] = -30
    coord_map[2, 0, :, :] = -40
    coord_map[2, -1, :, :] = -25
    inner_box = map_utils.inner_box(coord_map, box, stride=10)

    self.assertEqual(
        inner_box,
        bounding_box.BoundingBox(start=(100, 200, 196), size=(50, 50, 51)),
    )

  def test_invert_map(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    _, hx = np.mgrid[:50, :50]
    coord_map = np.zeros([2, 1, 50, 50])
    coord_map[1, 0, ...] = np.sin(hx / 25) * 20

    inv_map = map_utils.invert_map(coord_map, box, box, 40.0)

    np.testing.assert_array_almost_equal(
        inv_map[:, :, 1:, 1:], -coord_map[:, :, 1:, 1:], decimal=5
    )

  def test_invert_map_3d(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 5))
    _, _, hx = np.mgrid[:5, :50, :50]
    coord_map = np.zeros([3, 5, 50, 50])
    coord_map[1, ...] = np.sin(hx / 25) * 20

    inv_map = map_utils.invert_map(coord_map, box, box, 40.0)

    np.testing.assert_array_almost_equal(
        inv_map[:, 1:, 1:, 1:], -coord_map[:, 1:, 1:, 1:], decimal=5
    )

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
        resampled[:, :, :-1, :-1], expected[:, :, 6:-1, 4:-1], decimal=2
    )

  def test_compose_maps(self):
    box = bounding_box.BoundingBox(start=(100, 200, 10), size=(50, 50, 1))
    coord_map = np.zeros([2, 1, 50, 50])
    hy, hx = np.mgrid[:50, :50]
    coord_map[0, 0, ...] = np.sin(hx / 25)
    coord_map[1, 0, ...] = np.cos(hy / 25)
    stride = 5

    # Composing a map with its inversion should yield the identity transform.
    inverted = map_utils.invert_map(coord_map, box, box, stride)
    composed = map_utils.compose_maps(
        coord_map, box, stride, inverted, box, stride
    )[:, :, 1:-2, 1:-2]

    np.testing.assert_array_almost_equal(
        composed, np.zeros_like(composed), decimal=3
    )

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
        map_utils.compose_maps_fast(
            flow, box1.start[::-1], stride, coord_map, box2.start[::-1], stride
        )
    )
    np.testing.assert_array_equal(updated, flow)

    # Now test with a non-zero coordinate map.
    coord_map[0, :, :, 7:] = -10
    updated = np.array(
        map_utils.compose_maps_fast(
            flow, box1.start[::-1], stride, coord_map, box2.start[::-1], stride
        )
    )
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
        map_utils.compose_maps_fast(
            coord_map,
            box.start[::-1],
            stride,
            inverted,
            box.start[::-1],
            stride,
        )
    )[:, :, 1:-2, 1:-2]

    np.testing.assert_array_almost_equal(
        composed, np.zeros_like(composed), decimal=3
    )

  def test_mask_irregular(self):
    coord_map = np.zeros([2, 50, 50])
    coord_map[0, 40, 10] = 10
    bad = map_utils.mask_irregular(coord_map, (40, 40), 0.25, 1.1)

    expected = np.zeros([2, 50, 50])
    expected[:, 39:42, 8:11] = np.nan

    np.testing.assert_array_equal(expected, coord_map)
    np.testing.assert_array_equal(np.isnan(expected[0, ...]), bad)

  def _make_mixed_resample_map(self):
    src_box = bounding_box.BoundingBox(start=(100, 200, 5), size=(24, 20, 3))
    dst_box = bounding_box.BoundingBox(start=(196, 396, 5), size=(52, 44, 3))
    hy, hx = np.mgrid[:20, :24]
    cmap = np.zeros((2, 3, 20, 24), dtype=np.float32)
    for z in range(3):
      cmap[0, z] = np.sin((hx + z) / 7.0) * 5.0 + 0.3 * hy
      cmap[1, z] = np.cos((hy + z) / 6.0) * 4.0 - 0.2 * hx
    # Slice z=1 has NaNs to exercise mixed dense + Delaunay fallback.
    cmap[:, 1, 8:10, 9:11] = np.nan
    return cmap, src_box, dst_box

  def test_resample_map_linear_mixed_dense_and_nan_fallback(self):
    cmap, src_box, dst_box = self._make_mixed_resample_map()
    got_lin = map_utils.resample_map(
        cmap, src_box, dst_box, 40.0, 20.0, method='linear'
    )
    # Verify out-of-bounds margins are NaN and compare dense slices (z=0, 2)
    # against RegularGridInterpolator and fallback slice z=1 against
    # LinearNDInterpolator across both channels.
    self.assertTrue(np.all(np.isnan(got_lin[:, :, 0, :])))
    self.assertTrue(np.all(np.isnan(got_lin[:, :, :, 0])))

    src_y = (np.arange(20) + src_box.start[1]) * 40.0
    src_x = (np.arange(24) + src_box.start[0]) * 40.0
    qy, qx = np.mgrid[: dst_box.size[1], : dst_box.size[0]]
    pts_y = (qy + dst_box.start[1]) * 20.0
    pts_x = (qx + dst_box.start[0]) * 20.0
    sy_grid, sx_grid = np.meshgrid(src_y, src_x, indexing='ij')

    for z in range(3):
      for ch in range(2):
        with self.subTest(z=z, channel=ch):
          if z == 1:
            valid = np.isfinite(cmap[0, z])
            lin_nd = interpolate.LinearNDInterpolator(
                np.column_stack([sx_grid[valid], sy_grid[valid]]),
                cmap[ch, z][valid],
                fill_value=np.nan,
            )
            expected = lin_nd(pts_x, pts_y)
          else:
            rgi = interpolate.RegularGridInterpolator(
                (src_y, src_x),
                cmap[ch, z],
                method='linear',
                bounds_error=False,
            )
            expected = rgi((pts_y, pts_x))
          np.testing.assert_allclose(
              got_lin[ch, z], expected, rtol=1e-5, atol=1e-5, equal_nan=True
          )

  def test_resample_map_nearest_mixed_dense_and_nan_fallback(self):
    cmap, src_box, dst_box = self._make_mixed_resample_map()
    got_nn = map_utils.resample_map(
        cmap, src_box, dst_box, 40.0, 20.0, method='nearest'
    )

    src_y = (np.arange(20) + src_box.start[1]) * 40.0
    src_x = (np.arange(24) + src_box.start[0]) * 40.0
    qy, qx = np.mgrid[: dst_box.size[1], : dst_box.size[0]]
    pts_y = (qy + dst_box.start[1]) * 20.0
    pts_x = (qx + dst_box.start[0]) * 20.0
    sy_grid, sx_grid = np.meshgrid(src_y, src_x, indexing='ij')

    for z in range(3):
      for ch in range(2):
        with self.subTest(z=z, channel=ch):
          if z == 1:
            valid = np.isfinite(cmap[0, z])
            nn_nd = interpolate.NearestNDInterpolator(
                np.column_stack([sx_grid[valid], sy_grid[valid]]),
                cmap[ch, z][valid],
            )
            expected = nn_nd(pts_x, pts_y)
          else:
            rgi = interpolate.RegularGridInterpolator(
                (src_y, src_x),
                cmap[ch, z],
                method='nearest',
                bounds_error=False,
                fill_value=None,
            )
            expected = rgi((pts_y, pts_x))
          np.testing.assert_allclose(
              got_nn[ch, z], expected, rtol=1e-5, atol=1e-5
          )

  def test_compose_maps_large_float32_jit_and_sparse_fallback(self):
    rng = np.random.default_rng(2026)
    # Two dense 96x96 slices give 18,432 >= 16,384 nodes so
    # _compose_maps_dense_jit executes, while z=2 exercises sparse fallback.
    box1 = bounding_box.BoundingBox(start=(50000, 60000, 10), size=(96, 96, 3))
    box2 = bounding_box.BoundingBox(
        start=(49996, 59996, 10), size=(104, 104, 3)
    )
    m1 = rng.uniform(-15.0, 15.0, size=(2, 3, 96, 96)).astype(np.float32)
    m1[:, 0, 5, 5] = np.nan
    m1[0, 0, 0, 0] = -500.0  # out-of-bounds query into map2
    gy, gx = np.mgrid[:104, :104]
    m2 = np.zeros((2, 3, 104, 104), dtype=np.float32)
    for z in range(3):
      m2[0, z] = (0.25 * gx - 0.15 * gy + z).astype(np.float32)
      m2[1, z] = (-0.1 * gx + 0.2 * gy - z).astype(np.float32)
    m2[:, 2, 40:42, 40:42] = np.nan

    comp32 = map_utils.compose_maps(m1, box1, 40.0, m2, box2, 40.0)
    self.assertEqual(comp32.dtype, np.float32)
    self.assertTrue(np.all(np.isnan(comp32[:, 0, 5, 5])))
    self.assertTrue(np.all(np.isnan(comp32[:, 0, 0, 0])))

    iy, ix = np.mgrid[:96, :96]
    src_y2 = (np.arange(104, dtype=np.float64) + box2.start[1]) * 40.0
    src_x2 = (np.arange(104, dtype=np.float64) + box2.start[0]) * 40.0
    sy2_grid, sx2_grid = np.meshgrid(src_y2, src_x2, indexing='ij')

    for z in range(3):
      qx = (ix + box1.start[0]) * 40.0 + m1[0, z].astype(np.float64)
      qy = (iy + box1.start[1]) * 40.0 + m1[1, z].astype(np.float64)
      valid_q = np.isfinite(qx) & np.isfinite(qy)
      for ch in range(2):
        with self.subTest(z=z, channel=ch):
          expected = np.full((96, 96), np.nan, dtype=np.float64)
          if z < 2:
            rgi = interpolate.RegularGridInterpolator(
                (src_y2, src_x2),
                m2[ch, z].astype(np.float64),
                method='linear',
                bounds_error=False,
                fill_value=np.nan,
            )
            expected[valid_q] = m1[ch, z][valid_q] + rgi(
                (qy[valid_q], qx[valid_q])
            )
          else:
            valid_s = np.all(np.isfinite(m2[:, z]), axis=0)
            lin_nd = interpolate.LinearNDInterpolator(
                np.column_stack([sx2_grid[valid_s], sy2_grid[valid_s]]),
                m2[ch, z][valid_s].astype(np.float64),
                fill_value=np.nan,
            )
            expected[valid_q] = m1[ch, z][valid_q] + lin_nd(
                qx[valid_q], qy[valid_q]
            )
          np.testing.assert_allclose(
              comp32[ch, z], expected, rtol=1e-4, atol=1e-4, equal_nan=True
          )

  def test_invert_map_2d_large_jit_and_nan_fallback(self):
    # Large 2D invert_map (>= 16384 nodes -> JAX + float64 polish) on z=0
    # and Delaunay NaN fallback on z=1.
    src_box = bounding_box.BoundingBox(start=(50, 80, 0), size=(140, 130, 2))
    dst_box = bounding_box.BoundingBox(start=(49, 79, 0), size=(142, 132, 2))
    hy, hx = np.mgrid[:130, :140]
    cmap = np.zeros((2, 2, 130, 140), dtype=np.float64)
    cmap[0, :] = np.sin(hx / 30.0) * 8.0 + 0.05 * hy
    cmap[1, :] = np.cos(hy / 28.0) * 6.0 - 0.04 * hx
    cmap[:, 1, 60, 70] = np.nan  # triggers Delaunay fallback on z=1

    inv = map_utils.invert_map(cmap, src_box, dst_box, 20.0)
    self.assertEqual(inv.shape, (2, 2, 132, 142))
    self.assertTrue(np.all(np.isnan(inv[:, 0, 0, :])))

    src_z0 = src_box.adjusted_by(end=(0, 0, -1))
    dst_z0 = dst_box.adjusted_by(end=(0, 0, -1))
    comp_fwd_inv = map_utils.compose_maps(
        inv[:, 0:1], dst_z0, 20.0, cmap[:, 0:1], src_z0, 20.0
    )[:, 0, 2:-2, 2:-2]
    np.testing.assert_allclose(comp_fwd_inv, 0.0, atol=1e-4)
    comp_inv_fwd = map_utils.compose_maps(
        cmap[:, 0:1], src_z0, 20.0, inv[:, 0:1], dst_z0, 20.0
    )[:, 0, 2:-2, 2:-2]
    np.testing.assert_allclose(comp_inv_fwd, 0.0, atol=3e-3)

    # Verify recovered values on Delaunay fallback slice z=1 away from hole.
    comp_fb = map_utils.compose_maps(
        inv[:, 1:2], dst_z0, 20.0, cmap[:, 1:2], src_z0, 20.0
    )[:, 0, 2:-2, 2:-2]
    fb_interior = np.all(np.isfinite(comp_fb), axis=0)
    self.assertGreater(int(np.count_nonzero(fb_interior)), 15000)
    np.testing.assert_allclose(comp_fb[:, fb_interior], 0.0, atol=5e-3)

  def test_fill_missing_edt_non_rectangular_2d_and_3d(self):
    # 2D non-rectangular missing region hitting distance_transform_edt.
    hy, hx = np.mgrid[:12, :14]
    m2 = np.full((2, 1, 12, 14), np.nan, dtype=np.float64)
    valid_2d = ((hy - 6) ** 2 + (hx - 7) ** 2 <= 16) & ~((hy >= 6) & (hx >= 7))
    m2[0, 0, valid_2d] = hx[valid_2d] * 2.0 + hy[valid_2d] * 0.5
    m2[1, 0, valid_2d] = hy[valid_2d] * -3.0 + hx[valid_2d] * 0.25

    filled_2d = map_utils.fill_missing(
        m2, extrapolate=True, interpolate_first=False
    )
    self.assertTrue(np.all(np.isfinite(filled_2d)))
    vy2, vx2 = np.nonzero(valid_2d)
    for y, x in zip(*np.nonzero(~valid_2d)):
      sq_d = (vy2 - y) ** 2 + (vx2 - x) ** 2
      nearest = sq_d == sq_d.min()
      ny, nx = vy2[nearest], vx2[nearest]
      matches = (filled_2d[0, 0, y, x] == m2[0, 0, ny, nx]) & (
          filled_2d[1, 0, y, x] == m2[1, 0, ny, nx]
      )
      self.assertTrue(np.any(matches), msg=f'2D EDT mismatch at ({y}, {x})')

    # 3D non-rectangular missing region hitting distance_transform_edt.
    hz, hy3, hx3 = np.mgrid[:6, :8, :10]
    m3 = np.full((3, 6, 8, 10), np.nan, dtype=np.float32)
    valid_3d = (hz + hy3 + hx3 >= 6) & (hz <= 4) & (hy3 <= 6) & (hx3 <= 8)
    m3[0, valid_3d] = (hx3[valid_3d] * 1.5).astype(np.float32)
    m3[1, valid_3d] = (hy3[valid_3d] * -2.0).astype(np.float32)
    m3[2, valid_3d] = (hz[valid_3d] * 3.0).astype(np.float32)

    filled_3d_nr = map_utils.fill_missing(
        m3, extrapolate=True, interpolate_first=False
    )
    self.assertTrue(np.all(np.isfinite(filled_3d_nr)))
    vz3, vy3, vx3 = np.nonzero(valid_3d)
    for z, y, x in zip(*np.nonzero(~valid_3d)):
      sq_d = (vz3 - z) ** 2 + (vy3 - y) ** 2 + (vx3 - x) ** 2
      nearest = sq_d == sq_d.min()
      nz, ny, nx = vz3[nearest], vy3[nearest], vx3[nearest]
      matches = (
          (filled_3d_nr[0, z, y, x] == m3[0, nz, ny, nx])
          & (filled_3d_nr[1, z, y, x] == m3[1, nz, ny, nx])
          & (filled_3d_nr[2, z, y, x] == m3[2, nz, ny, nx])
      )
      self.assertTrue(
          np.any(matches), msg=f'3D EDT mismatch at ({z}, {y}, {x})'
      )

  def test_fill_missing_solid_box_3d(self):
    m3 = np.ones((3, 8, 10, 12), dtype=np.float32) * 3.0
    m3[:, :2, :, :] = np.nan
    m3[:, -1:, :, :] = np.nan
    m3[:, :, :1, :] = np.nan
    self.assertTrue(np.isnan(map_utils.fill_missing(m3)[0, 0, 5, 5]))
    np.testing.assert_allclose(
        map_utils.fill_missing(m3, extrapolate=True), 3.0
    )

  def test_compose_maps_fast_start2_gt_start1_and_boundary_modes(self):
    rng = np.random.default_rng(42)
    # 2D multi-Z with start2 > start1 and non-constant maps across all 5 modes.
    start1_2d = (0, 1, 2)
    start2_2d = (0, 4, 6)
    stride_2d = (20.0, 40.0)
    map1_2d = rng.uniform(-60.0, 180.0, size=(2, 4, 12, 14)).astype(np.float32)
    map1_2d[:, 1, 3, 5] = np.nan
    gy2, gx2 = np.mgrid[:16, :18]
    map2_2d = np.zeros((2, 4, 16, 18), dtype=np.float32)
    for z in range(4):
      map2_2d[0, z] = (0.4 * gx2 - 0.25 * gy2 + z * 3.0).astype(np.float32)
      map2_2d[1, z] = (-0.15 * gx2 + 0.35 * gy2 - z * 2.0).astype(np.float32)

    scipy_modes = {
        'nearest': 'nearest',
        'constant': 'grid-constant',
        'wrap': 'grid-wrap',
        'mirror': 'mirror',
        'reflect': 'reflect',
    }
    iy1, ix1 = np.mgrid[:12, :14]
    ref1_x = (ix1 + start1_2d[2]) * stride_2d[1]
    ref1_y = (iy1 + start1_2d[1]) * stride_2d[0]
    ref2_x = (gx2 + start2_2d[2]) * stride_2d[1]
    ref2_y = (gy2 + start2_2d[1]) * stride_2d[0]
    off2_x = start2_2d[2] * stride_2d[1]
    off2_y = start2_2d[1] * stride_2d[0]

    for mode, sp_mode in scipy_modes.items():
      got = np.asarray(
          map_utils.compose_maps_fast(
              map1_2d,
              start1_2d,
              stride_2d,
              map2_2d,
              start2_2d,
              stride_2d,
              mode=mode,
          )
      )
      for z in range(4):
        with self.subTest(dim=2, mode=mode, z=z):
          qx = (ref1_x + map1_2d[0, z] - off2_x) / stride_2d[1]
          qy = (ref1_y + map1_2d[1, z] - off2_y) / stride_2d[0]
          abs2_x = map2_2d[0, z] + ref2_x
          abs2_y = map2_2d[1, z] + ref2_y
          exp_x = (
              ndimage.map_coordinates(
                  abs2_x, [qy, qx], order=1, mode=sp_mode, cval=np.nan
              )
              - ref1_x
          )
          exp_y = (
              ndimage.map_coordinates(
                  abs2_y, [qy, qx], order=1, mode=sp_mode, cval=np.nan
              )
              - ref1_y
          )
          np.testing.assert_allclose(
              got[:, z],
              np.stack([exp_x, exp_y], axis=0),
              rtol=1e-4,
              atol=1e-3,
              equal_nan=True,
          )

    # 3D non-trivial composition with start2 > start1 and anisotropic stride.
    start1_3d = (1, 2, 3)
    start2_3d = (3, 4, 6)
    stride_3d = (5.0, 10.0, 20.0)
    map1_3d = rng.uniform(-15.0, 60.0, size=(3, 6, 8, 9)).astype(np.float32)
    gz3, gy3, gx3 = np.mgrid[:8, :10, :12]
    map2_3d = np.stack(
        [
            0.3 * gx3 - 0.2 * gy3 + 0.1 * gz3,
            -0.1 * gx3 + 0.25 * gy3 - 0.15 * gz3,
            0.05 * gx3 + 0.1 * gy3 + 0.2 * gz3,
        ],
        axis=0,
    ).astype(np.float32)
    iz1, iy1_3d, ix1_3d = np.mgrid[:6, :8, :9]
    r1x = (ix1_3d + start1_3d[2]) * stride_3d[2]
    r1y = (iy1_3d + start1_3d[1]) * stride_3d[1]
    r1z = (iz1 + start1_3d[0]) * stride_3d[0]
    r2x = (gx3 + start2_3d[2]) * stride_3d[2]
    r2y = (gy3 + start2_3d[1]) * stride_3d[1]
    r2z = (gz3 + start2_3d[0]) * stride_3d[0]

    for mode, sp_mode in scipy_modes.items():
      with self.subTest(dim=3, mode=mode):
        got_3d = np.asarray(
            map_utils.compose_maps_fast(
                map1_3d,
                start1_3d,
                stride_3d,
                map2_3d,
                start2_3d,
                stride_3d,
                mode=mode,
            )
        )
        qx = (r1x + map1_3d[0] - start2_3d[2] * stride_3d[2]) / stride_3d[2]
        qy = (r1y + map1_3d[1] - start2_3d[1] * stride_3d[1]) / stride_3d[1]
        qz = (r1z + map1_3d[2] - start2_3d[0] * stride_3d[0]) / stride_3d[0]
        exp_3d = np.stack(
            [
                ndimage.map_coordinates(
                    map2_3d[0] + r2x,
                    [qz, qy, qx],
                    order=1,
                    mode=sp_mode,
                    cval=np.nan,
                )
                - r1x,
                ndimage.map_coordinates(
                    map2_3d[1] + r2y,
                    [qz, qy, qx],
                    order=1,
                    mode=sp_mode,
                    cval=np.nan,
                )
                - r1y,
                ndimage.map_coordinates(
                    map2_3d[2] + r2z,
                    [qz, qy, qx],
                    order=1,
                    mode=sp_mode,
                    cval=np.nan,
                )
                - r1z,
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            got_3d, exp_3d, rtol=1e-4, atol=1e-3, equal_nan=True
        )

  def test_mask_irregular_batch_and_processor(self):
    irr_batch = np.zeros((2, 3, 20, 20), dtype=np.float32)
    irr_batch[0, 0, 10, 5] = 25.0
    irr_batch[1, 2, 6, 12] = -30.0
    irr_ref = irr_batch.copy()
    bad_b = map_utils.mask_irregular_batch(irr_batch, (20.0, 20.0), 0.25, 1.1)
    for z in range(3):
      with self.subTest(z=z):
        bad_z = map_utils.mask_irregular(irr_ref[:, z], (20.0, 20.0), 0.25, 1.1)
        np.testing.assert_array_equal(bad_b[z], bad_z)
    np.testing.assert_array_equal(irr_batch, irr_ref)

    proc = maps_proc.MaskIrregularities(stride=(20.0, 20.0), frac=0.25)
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=(20, 20, 3))
    proc.set_effective_subvol_and_overlap(box.size, proc.overlap())
    out_subvol = proc.process(subvolume.Subvolume(irr_ref, box))
    self.assertEqual(
        out_subvol.bbox,
        bounding_box.BoundingBox(start=(3, 3, 0), size=(14, 14, 3)),
    )
    np.testing.assert_array_equal(out_subvol.data, irr_batch[:, :, 3:-3, 3:-3])

  def test_reconcile_cross_block_maps_process_and_fallback(self):
    gy, gx = np.mgrid[:24, :24]
    # Non-constant main volume covering z=0..10 (2 blocks: [0..5] and [5..10]).
    base_data = np.zeros((2, 11, 24, 24), dtype=np.float32)
    for z in range(11):
      base_data[0, z] = 0.1 * gx + 0.05 * gy + 0.2 * z
      base_data[1, z] = -0.1 * gy + 0.04 * gx + 0.1 * z

    def _mk_vol(val_x, val_y, nz=3):
      vol = np.zeros((2, nz, 24, 24), dtype=np.float32)
      for z in range(nz):
        vol[0, z] = val_x + 0.03 * gx - 0.02 * gy + 0.1 * z
        vol[1, z] = val_y - 0.02 * gx + 0.03 * gy - 0.1 * z
      return vol

    xb = _mk_vol(1.5, -0.5)
    last_inv = np.zeros((2, 11, 24, 24), dtype=np.float32)
    main_inv = np.zeros((2, 11, 24, 24), dtype=np.float32)
    for z in range(11):
      last_inv[0, z] = -0.05 * gx + 0.2
      last_inv[1, z] = 0.05 * gy - 0.1
      main_inv[0, z] = -0.08 * gx + 0.5
      main_inv[1, z] = 0.08 * gy - 0.3

    box_all = bounding_box.BoundingBox(start=(0, 0, 0), size=(24, 24, 11))
    xb_inv = map_utils.invert_map(
        xb,
        bounding_box.BoundingBox(start=(0, 0, 0), size=(24, 24, 3)),
        bounding_box.BoundingBox(start=(0, 0, 0), size=(24, 24, 3)),
        20.0,
    )
    volumes = {
        'xb': xb,
        'xb_inv': xb_inv,
        'last_inv': last_inv,
        'main_inv': main_inv,
    }

    for backward in (False, True):
      with self.subTest(backward=backward):
        cfg = maps_proc.ReconcileCrossBlockMaps.Config(
            cross_block='xb',
            cross_block_inv='xb_inv',
            last_inv='last_inv',
            main_inv='main_inv',
            z_map={'0': 0, '5': 1, '10': 2},
            stride=20,
            xy_overlap=0,
            backward=backward,
        )
        proc = _ReconcileWithMemVolumes(cfg, volumes)
        proc.set_effective_subvol_and_overlap(box_all.size, proc.overlap())
        out = proc.process(subvolume.Subvolume(base_data.copy(), box_all))
        self.assertEqual(
            out.bbox,
            bounding_box.BoundingBox(start=(0, 0, 1), size=(24, 24, 10)),
        )

        # Independently verify block-boundary section z=5 and final section z=10
        # match expected boundary maps.
        if backward:
          np.testing.assert_allclose(out.data[:, 4], xb[:, 1], atol=1e-5)
          np.testing.assert_allclose(out.data[:, 9], 0.0, atol=1e-5)
        else:
          np.testing.assert_allclose(out.data[:, 4], xb[:, 1], atol=1e-5)
          np.testing.assert_allclose(out.data[:, 9], xb[:, 2], atol=1e-5)

        # Verify interior section z=7 matches independent single-section
        # composition (section * xblock_pre) * scaled_offset.
        flat_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(24, 24, 1))
        if backward:
          xb_pre = np.zeros_like(xb[:, 1:2])
          xb_pre_inv = np.zeros_like(xb_pre)
          xb_post = xb[:, 1:2]
          end_inv = last_inv[:, 5:6]
          scale = (10 - 7) / 5.0
        else:
          xb_pre = xb[:, 1:2]
          xb_pre_inv = xb_inv[:, 1:2]
          xb_post = xb[:, 2:3]
          end_inv = main_inv[:, 10:11]
          scale = (7 - 5) / 5.0
        offset = map_utils.compose_maps(
            map_utils.compose_maps(
                xb_pre_inv, flat_box, 20, end_inv, flat_box, 20
            ),
            flat_box,
            20,
            xb_post,
            flat_box,
            20,
        )
        expected_z7 = map_utils.compose_maps(
            map_utils.compose_maps(
                base_data[:, 7:8], flat_box, 20, xb_pre, flat_box, 20
            ),
            flat_box,
            20,
            offset * scale,
            flat_box,
            20,
        )
        np.testing.assert_allclose(
            out.data[:, 6:7], expected_z7, rtol=1e-5, atol=1e-5
        )

    # Verify QhullError fallback in ReconcileCrossBlockMaps._interpolate.
    cfg_fb = maps_proc.ReconcileCrossBlockMaps.Config(
        cross_block='xb',
        cross_block_inv='xb_inv',
        last_inv='last_inv',
        main_inv='main_inv',
        z_map={'0': 0, '5': 1, '10': 2},
        stride=20,
        xy_overlap=0,
        backward=False,
    )
    proc_fb = _ReconcileWithMemVolumes(cfg_fb, volumes)
    box_b2 = bounding_box.BoundingBox(start=(0, 0, 5), size=(24, 24, 6))
    proc_fb.set_effective_subvol_and_overlap(box_b2.size, proc_fb.overlap())
    got_batched = proc_fb.process(
        subvolume.Subvolume(base_data[:, 5:11].copy(), box_b2)
    )
    orig_compose = map_utils.compose_maps
    call_count = 0

    def _raise_on_batch_compose(m1, *args, **kwargs):
      nonlocal call_count
      call_count += 1
      # Call 1-2 compute `offset`; call 3 is the batched interior composition.
      if call_count == 3 and m1.shape[1] > 1:
        raise spatial.qhull.QhullError('simulated batch QhullError')
      return orig_compose(m1, *args, **kwargs)

    with mock.patch.object(
        map_utils, 'compose_maps', side_effect=_raise_on_batch_compose
    ):
      got_fallback = proc_fb.process(
          subvolume.Subvolume(base_data[:, 5:11].copy(), box_b2)
      )
    self.assertGreaterEqual(call_count, 3)
    np.testing.assert_allclose(
        got_fallback.data, got_batched.data, rtol=1e-5, atol=1e-5
    )

  def test_invert_map_coupled_padded_and_fallback(self):
    src_box = bounding_box.BoundingBox(start=(10, 20, 5), size=(40, 35, 2))
    dst_box = src_box.adjusted_by(start=(-2, -2, 0), end=(2, 2, 0))
    hy, hx = np.mgrid[:35, :40]
    coord_map = np.zeros([2, 2, 35, 40])
    coord_map[0, 0] = 350.0 + 6.0 * np.sin(hx / 12.0) * np.cos(hy / 15.0)
    coord_map[1, 0] = -120.0 + 5.0 * np.cos(hx / 14.0) * np.sin(hy / 11.0)
    # Section 1 has an interior NaN hole to exercise Qhull fallback.
    coord_map[:, 1] = coord_map[:, 0] * 0.05
    coord_map[:, 1, 15:18, 18:21] = np.nan

    # dst_box shifted by mean displacement of section 0.
    out_box = map_utils.outer_box(
        coord_map[:, 0:1], src_box.adjusted_by(end=(0, 0, -1)), stride=40.0
    )
    inv0 = map_utils.invert_map(
        coord_map[:, 0:1],
        src_box.adjusted_by(end=(0, 0, -1)),
        out_box,
        40.0,
    )
    # Exact inverse check: coord_map(inv0(q)) == q to float32 precision.
    composed0_rev = np.array(
        map_utils.compose_maps_fast(
            inv0,
            out_box.start[::-1],
            40.0,
            coord_map[:, 0:1],
            src_box.adjusted_by(end=(0, 0, -1)).start[::-1],
            40.0,
            mode='constant',
        )
    )
    finite0 = np.isfinite(composed0_rev)
    self.assertTrue(np.any(finite0))
    np.testing.assert_allclose(composed0_rev[finite0], 0.0, atol=5e-4)

    composed0_fast = np.array(
        map_utils.compose_maps_fast(
            coord_map[:, 0:1],
            src_box.adjusted_by(end=(0, 0, -1)).start[::-1],
            40.0,
            inv0,
            out_box.start[::-1],
            40.0,
            mode='constant',
        )
    )[:, :, 2:-2, 2:-2]
    np.testing.assert_allclose(composed0_fast, 0.0, atol=1.5e-2)

    composed0 = map_utils.compose_maps(
        coord_map[:, 0:1],
        src_box.adjusted_by(end=(0, 0, -1)),
        40.0,
        inv0,
        out_box,
        40.0,
    )[:, :, 2:-2, 2:-2]
    np.testing.assert_allclose(composed0, np.zeros_like(composed0), atol=1.5e-2)

    inv_both = map_utils.invert_map(coord_map, src_box, dst_box, 40.0)
    self.assertEqual(inv_both.shape, (2, 2, 39, 44))
    comp_fb = map_utils.compose_maps(
        inv_both[:, 1:2],
        dst_box.adjusted_by(end=(0, 0, -1)),
        40.0,
        coord_map[:, 1:2],
        src_box.adjusted_by(end=(0, 0, -1)),
        40.0,
    )
    valid_fb = np.all(np.isfinite(comp_fb), axis=0)
    self.assertGreater(int(np.count_nonzero(valid_fb)), 500)
    np.testing.assert_allclose(comp_fb[:, valid_fb], 0.0, atol=5e-3)

  def test_invert_map_3d_coupled(self):
    box = bounding_box.BoundingBox(start=(10, 20, 30), size=(16, 14, 8))
    hz, hy, hx = np.mgrid[:8, :14, :16]
    coord_map = np.zeros([3, 8, 14, 16])
    coord_map[0] = 80.0 + 3.0 * np.sin(hx / 6.0) * np.cos(hy / 7.0)
    coord_map[1] = -40.0 + 2.5 * np.cos(hy / 6.0) * np.sin(hz / 4.0)
    coord_map[2] = 20.0 + 2.0 * np.sin(hz / 4.0) * np.cos(hx / 8.0)
    out_box = map_utils.outer_box(coord_map, box, stride=20.0)
    inv = map_utils.invert_map(coord_map, box, out_box, 20.0)
    composed_rev = np.array(
        map_utils.compose_maps_fast(
            inv,
            out_box.start[::-1],
            20.0,
            coord_map,
            box.start[::-1],
            20.0,
            mode='constant',
        )
    )
    finite3d = np.isfinite(composed_rev)
    self.assertTrue(np.any(finite3d))
    np.testing.assert_allclose(composed_rev[finite3d], 0.0, atol=5e-4)

    composed = np.array(
        map_utils.compose_maps_fast(
            coord_map,
            box.start[::-1],
            20.0,
            inv,
            out_box.start[::-1],
            20.0,
            mode='constant',
        )
    )[:, 2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(composed, 0.0, atol=2.5e-2)

  def test_fill_missing_padded_box_and_3d(self):
    hy, hx = np.mgrid[:20, :24]
    padded = np.full((2, 2, 20, 24), np.nan)
    padded[0, :, 2:-2, 3:-3] = hx[2:-2, 3:-3] * 0.5
    padded[1, :, 2:-2, 3:-3] = hy[2:-2, 3:-3] * -0.25
    filled = map_utils.fill_missing(padded, extrapolate=True)
    self.assertFalse(np.any(np.isnan(filled)))
    np.testing.assert_array_equal(filled[0, 0, 0, 3:-3], padded[0, 0, 2, 3:-3])
    np.testing.assert_array_equal(
        filled[1, 1, -1, 3:-3], padded[1, 1, -3, 3:-3]
    )

    map_3d = np.full((3, 6, 10, 12), np.nan)
    map_3d[:, 1:-1, 2:-2, 2:-2] = 3.5
    filled_3d = map_utils.fill_missing(
        map_3d, extrapolate=True, interpolate_first=False
    )
    np.testing.assert_allclose(filled_3d, 3.5)

  def test_compose_maps_partial_nan_border_holes_and_cache(self):
    rng = np.random.default_rng(77)
    # Large float32 batch (>= 16384 nodes) with curved outer NaN border,
    # interior holes, concave notch, and both shared and distinct masks across z
    # to exercise JIT bilinear, boundary Delaunay, and cache reuse/invalidation.
    nz, h, w = 4, 80, 96
    box = bounding_box.BoundingBox(start=(1000, 2000, 0), size=(w, h, nz))
    gy, gx = np.mgrid[:h, :w]
    m1 = np.zeros((2, nz, h, w), dtype=np.float32)
    m2 = np.zeros((2, nz, h, w), dtype=np.float32)
    for z in range(nz):
      m1[0, z] = (
          rng.uniform(-25.0, 25.0, size=(h, w)) + 0.1 * gx
      ).astype(np.float32)
      m1[1, z] = (
          rng.uniform(-25.0, 25.0, size=(h, w)) - 0.1 * gy
      ).astype(np.float32)
      m2[0, z] = (0.3 * gx - 0.2 * gy + 2.0 * z).astype(np.float32)
      m2[1, z] = (-0.15 * gx + 0.25 * gy - 1.5 * z).astype(np.float32)

    # Slices 0 and 1 share an identical curved border + concave notch + holes.
    border_invalid = (
        (gx < (3 + 2 * np.sin(gy / 10.0)))
        | (gx > (w - 4 - 2 * np.cos(gy / 12.0)))
        | (gy < 3)
        | (gy > h - 4)
    )
    notch = (gy < 12) & (gx >= 35) & (gx < 45)
    hole1 = (gy >= 30) & (gy < 34) & (gx >= 40) & (gx < 46)
    slit = (gy == 55) & (gx >= 20) & (gx < 35)
    mask01 = border_invalid | notch | hole1 | slit
    m2[:, 0, mask01] = np.nan
    m2[:, 1, mask01] = np.nan
    # Slice 2 has a different hole pattern (cache miss).
    mask2 = border_invalid | ((gy >= 40) & (gy < 45) & (gx >= 60) & (gx < 65))
    m2[:, 2, mask2] = np.nan
    # Slice 3 is all NaN.
    m2[:, 3, :, :] = np.nan

    orig_delaunay = spatial.Delaunay
    delaunay_point_counts: list[int] = []

    def _counting_delaunay(pts, *args, **kwargs):
      delaunay_point_counts.append(len(pts))
      return orig_delaunay(pts, *args, **kwargs)

    with mock.patch.object(
        spatial, 'Delaunay', side_effect=_counting_delaunay
    ):
      got = map_utils.compose_maps(m1, box, 40.0, m2, box, 40.0)

    # Only 2 triangulations built (shared mask for z=0,1 + distinct mask z=2),
    # each using only boundary points (<< total valid points).
    self.assertLen(delaunay_point_counts, 2)
    self.assertLess(max(delaunay_point_counts), 800)
    self.assertTrue(np.all(np.isnan(got[:, 3])))

    for z in range(3):
      qx = gx + m1[0, z].astype(np.float64) / 40.0
      qy = gy + m1[1, z].astype(np.float64) / 40.0
      valid_s = np.all(np.isfinite(m2[:, z]), axis=0)
      pts_s = np.column_stack([gx[valid_s], gy[valid_s]])
      for ch in range(2):
        with self.subTest(z=z, ch=ch):
          lin_nd = interpolate.LinearNDInterpolator(
              pts_s, m2[ch, z][valid_s].astype(np.float64), fill_value=np.nan
          )
          expected = m1[ch, z].astype(np.float64) + lin_nd(qx, qy)
          np.testing.assert_allclose(
              got[ch, z], expected, rtol=1e-4, atol=1e-3, equal_nan=True
          )

    # Solid inner-box valid region (outer NaN border only) skips Qhull entirely
    # when residual queries stay outside the valid box, and 1-pixel-thin map2
    # returns all-NaN cleanly.
    m2_box_nan = np.ones((2, 1, 20, 24), dtype=np.float32) * 4.0
    m2_box_nan[:, :, :2, :] = np.nan
    m2_box_nan[:, :, -2:, :] = np.nan
    m2_box_nan[:, :, :, :2] = np.nan
    m2_box_nan[:, :, :, -2:] = np.nan
    m1_shift = np.full((2, 1, 20, 24), -10.0, dtype=np.float32)
    b_small = bounding_box.BoundingBox(start=(0, 0, 0), size=(24, 20, 1))
    delaunay_point_counts.clear()
    with mock.patch.object(
        spatial, 'Delaunay', side_effect=_counting_delaunay
    ):
      got_box = map_utils.compose_maps(
          m1_shift, b_small, 40.0, m2_box_nan, b_small, 40.0
      )
    self.assertEmpty(delaunay_point_counts)
    np.testing.assert_allclose(got_box[:, 0, 3:-2, 3:-2], -6.0)
    self.assertTrue(np.all(np.isnan(got_box[:, 0, :2, :])))

    b_thin = bounding_box.BoundingBox(start=(0, 0, 0), size=(1, 20, 1))
    m2_thin = np.zeros((2, 1, 20, 1), dtype=np.float32)
    got_thin = map_utils.compose_maps(
        m1_shift, b_small, 40.0, m2_thin, b_thin, 40.0
    )
    self.assertTrue(np.all(np.isnan(got_thin)))

  def test_invert_map_partial_nan_curved_border_and_hole(self):
    h, w = 64, 72
    box = bounding_box.BoundingBox(start=(10, 20, 0), size=(w, h, 1))
    gy, gx = np.mgrid[:h, :w]
    cmap = np.zeros((2, 1, h, w), dtype=np.float64)
    cmap[0, 0] = 25.0 + 6.0 * np.sin(gx / 18.0) + 0.08 * gy
    cmap[1, 0] = -15.0 + 5.0 * np.cos(gy / 16.0) - 0.06 * gx
    # Curved outer border NaN strip + interior 3x3 hole.
    invalid = (
        (gx < (3 + 1.5 * np.sin(gy / 9.0)))
        | (gx > (w - 4))
        | (gy < 2)
        | (gy > h - 3)
        | ((gy >= 28) & (gy < 31) & (gx >= 32) & (gx < 35))
    )
    cmap[:, 0, invalid] = np.nan

    inv = map_utils.invert_map(cmap, box, box, 40.0)
    # Compare against full Delaunay reference on finite entries and hull mask.
    valid_s = np.all(np.isfinite(cmap[:, 0]), axis=0)
    bu = gx[valid_s] + cmap[0, 0, valid_s] / 40.0
    bv = gy[valid_s] + cmap[1, 0, valid_s] / 40.0
    ref_u = interpolate.LinearNDInterpolator(
        np.column_stack([bu, bv]), -cmap[0, 0, valid_s], fill_value=np.nan
    )(gx.astype(np.float64), gy.astype(np.float64))
    ref_v = interpolate.LinearNDInterpolator(
        np.column_stack([bu, bv]), -cmap[1, 0, valid_s], fill_value=np.nan
    )(gx.astype(np.float64), gy.astype(np.float64))

    both_finite = np.isfinite(inv[0, 0]) & np.isfinite(ref_u)
    self.assertGreater(int(np.count_nonzero(both_finite)), 3000)
    np.testing.assert_allclose(
        inv[0, 0, both_finite], ref_u[both_finite], atol=2e-2
    )
    np.testing.assert_allclose(
        inv[1, 0, both_finite], ref_v[both_finite], atol=2e-2
    )

    # Exact forward-inverse composition check on interior valid region.
    comp = map_utils.compose_maps(inv, box, 40.0, cmap, box, 40.0)
    valid_c = np.all(np.isfinite(comp[:, 0]), axis=0)
    self.assertGreater(int(np.count_nonzero(valid_c)), 3000)
    np.testing.assert_allclose(comp[:, 0, valid_c], 0.0, atol=2e-2)

  def test_invert_map_3d_partial_nan_jit_and_fold_fallback(self):
    # 3D >= 65536 dst nodes (18x60x62 = 66,960) with anisotropic stride, shifted
    # dst_box, outer NaN border, and interior NaN hole to exercise 3D JIT +
    # _invert_partial_newton.
    src_box = bounding_box.BoundingBox(start=(4, 8, 2), size=(60, 58, 18))
    dst_box = src_box.adjusted_by(start=(-1, -1, 0), end=(1, 1, 0))
    gz, gy, gx = np.mgrid[:18, :58, :60]
    stride = (10.0, 15.0, 20.0)
    cmap = np.zeros((3, 18, 58, 60), dtype=np.float64)
    cmap[0] = 12.0 + 2.5 * np.sin(gx / 9.0) * np.cos(gy / 10.0)
    cmap[1] = -9.0 + 2.0 * np.cos(gy / 8.0) * np.sin(gz / 5.0)
    cmap[2] = 4.0 + 1.5 * np.sin(gz / 5.0) * np.cos(gx / 11.0)
    cmap[:, 0, :, :] = np.nan
    cmap[:, :, :2, :] = np.nan
    cmap[:, 6:9, 24:27, 25:28] = np.nan

    inv = map_utils.invert_map(cmap, src_box, dst_box, stride)
    self.assertEqual(inv.shape, (3, 18, 60, 62))
    comp_rev = np.asarray(
        map_utils.compose_maps_fast(
            inv,
            dst_box.start[::-1],
            stride,
            cmap,
            src_box.start[::-1],
            stride,
            mode='constant',
        )
    )
    # Interior valid cells should invert to exact trilinear precision, while
    # hole cells smoothly interpolate via 3D boundary Delaunay.
    interior_ok = np.all(np.isfinite(comp_rev), axis=0)
    self.assertGreater(int(np.count_nonzero(interior_ok)), 10000)
    np.testing.assert_allclose(comp_rev[:, interior_ok], 0.0, atol=2e-2)

    # Non-injective folds in both 2D and 3D fall back to full Delaunay.
    fold_2d = np.zeros((2, 2, 10, 12), dtype=np.float64)
    fold_2d[0, 0] = 4.0
    fold_2d[0, 1] = 4.0
    fold_2d[0, 1, 5, 5] = 80.0
    box_2d = bounding_box.BoundingBox(start=(0, 0, 0), size=(12, 10, 2))
    inv_fold_2d = map_utils.invert_map(fold_2d, box_2d, box_2d, 20.0)
    np.testing.assert_allclose(inv_fold_2d[0, 0, :, 1:], -4.0, atol=1e-6)
    self.assertTrue(np.isfinite(inv_fold_2d[0, 1, 1, 2]))

    fold_3d = np.zeros((3, 6, 8, 9), dtype=np.float64)
    fold_3d[0] = 4.0
    fold_3d[0, 3, 4, 4] = 80.0
    box_3d = bounding_box.BoundingBox(start=(0, 0, 0), size=(9, 8, 6))
    inv_fold_3d = map_utils.invert_map(fold_3d, box_3d, box_3d, 20.0)
    self.assertEqual(inv_fold_3d.shape, (3, 6, 8, 9))
    self.assertTrue(np.isfinite(inv_fold_3d[0, 1, 1, 2]))

  def test_fill_missing_2d_boundary_matches_full_delaunay(self):
    h, w = 48, 56
    gy, gx = np.mgrid[:h, :w]
    cmap = np.zeros((2, 1, h, w), dtype=np.float64)
    cmap[0, 0] = 0.4 * gx - 0.2 * gy + 3.0
    cmap[1, 0] = -0.3 * gx + 0.5 * gy - 2.0
    missing = (
        (gx < (4 + 2 * np.sin(gy / 8.0)))
        | (gy >= 20) & (gy < 24) & (gx >= 24) & (gx < 28)
    )
    cmap[:, 0, missing] = np.nan

    filled = map_utils.fill_missing(cmap, extrapolate=False)
    valid = ~missing
    pts = np.column_stack([gx[valid], gy[valid]])
    for ch in range(2):
      with self.subTest(ch=ch):
        expected = cmap[ch, 0].copy()
        expected[missing] = interpolate.LinearNDInterpolator(
            pts, cmap[ch, 0, valid], fill_value=np.nan
        )(gx[missing], gy[missing])
        np.testing.assert_allclose(
            filled[ch, 0], expected, rtol=1e-6, atol=1e-6, equal_nan=True
        )


if __name__ == '__main__':
  absltest.main()
