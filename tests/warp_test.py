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

"""Tests for warp."""

from unittest import mock
from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.volume import subvolume
import numpy as np
from scipy import interpolate
from scipy import ndimage
from sofima import map_utils
from sofima import warp
from sofima.processor import warp as warp_processor


def _ref_warp_subvolume_rgi(
    image: np.ndarray,
    image_box: bounding_box.BoundingBox,
    coord_map: np.ndarray,
    map_box: bounding_box.BoundingBox,
    stride: float,
    out_box: bounding_box.BoundingBox,
    interpolation: int,
    offset: float = 0.0,
) -> np.ndarray:
  abs_map = map_utils.to_absolute(coord_map, stride)
  abs_map += (
      map_box.start[:2] * stride - image_box.start[:2] + offset
  ).reshape(2, 1, 1, 1)
  hm, wm = coord_map.shape[2], coord_map.shape[3]
  map_y, map_x = np.ogrid[:hm, :wm]
  map_y = (map_y + map_box.start[1]) * stride - out_box.start[1] + offset
  map_x = (map_x + map_box.start[0]) * stride - out_box.start[0] + offset
  map_points = (map_y.ravel(), map_x.ravel())
  out_y, out_x = np.mgrid[: out_box.size[1], : out_box.size[0]]
  ref = np.zeros(
      (image.shape[0], out_box.size[2], out_box.size[1], out_box.size[0]),
      dtype=image.dtype,
  )
  for z in range(out_box.size[2]):
    dx = interpolate.RegularGridInterpolator(
        map_points, abs_map[0, z], bounds_error=False, fill_value=None
    )((out_y, out_x)).astype(np.float32)
    dy = interpolate.RegularGridInterpolator(
        map_points, abs_map[1, z], bounds_error=False, fill_value=None
    )((out_y, out_x)).astype(np.float32)
    dx_cv, dy_cv = warp.cv.convertMaps(
        dx, dy, dstmap1type=warp.cv.CV_16SC2, nninterpolation=False
    )
    for c in range(image.shape[0]):
      ref[c, z] = warp.cv.remap(
          image[c, z], dx_cv, dy_cv, interpolation=interpolation
      )
  return ref


def _ref_warp_points_rgi(
    points: np.ndarray,
    coord_map: np.ndarray,
    map_box: bounding_box.BoundingBox,
    stride: float,
) -> np.ndarray:
  hm, wm = coord_map.shape[2], coord_map.shape[3]
  abs_p = map_utils.to_absolute(coord_map, stride)
  abs_p += np.array(map_box.start[:2] * stride).reshape((2, 1, 1, 1))
  iiy, iix = np.ogrid[:hm, :wm]
  src_p = (
      ((iiy + map_box.start[1]) * stride).ravel(),
      ((iix + map_box.start[0]) * stride).ravel(),
  )
  ref = points.copy()
  for z_val in np.unique(points[:, 2]):
    sel = points[:, 2] == z_val
    zr = int(z_val - map_box.start[2])
    rx = interpolate.RegularGridInterpolator(
        src_p, abs_p[0, zr], bounds_error=False, fill_value=None
    )((points[sel, 1], points[sel, 0])).astype(np.float32)
    ry = interpolate.RegularGridInterpolator(
        src_p, abs_p[1, zr], bounds_error=False, fill_value=None
    )((points[sel, 1], points[sel, 0])).astype(np.float32)
    if np.issubdtype(ref.dtype, np.integer):
      ref[sel, 0] = np.round(rx).astype(ref.dtype)
      ref[sel, 1] = np.round(ry).astype(ref.dtype)
    else:
      ref[sel, 0] = rx
      ref[sel, 1] = ry
  return ref


class WarpTest(absltest.TestCase):

  def test_warp_subvolume_segmentation_translate(self):
    image = np.zeros((1, 2, 100, 100), dtype=np.uint64)
    image[0, 0, 40, 30] = 42
    image[0, 1, 50, 40] = 2**40
    image_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(100, 100, 2))

    # Coord map is larger than the requested output.
    coord_map = np.zeros((2, 2, 15, 15))
    coord_map[0, 0, :, :] = 10
    coord_map[1, 1, :, :] = 17
    map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(15, 15, 2))
    edge_len = 10

    # Output box at an offset relative to the input.
    out_box = bounding_box.BoundingBox(start=(10, 20, 0), size=(90, 80, 2))

    warped = warp.warp_subvolume(image, image_box, coord_map, map_box, edge_len,
                                 out_box)

    expected = np.zeros((1, 2, 80, 90))
    expected[0, 0, 20, 10] = 42
    expected[0, 1, 13, 30] = 2**40

    np.testing.assert_array_equal(warped, expected)

  def test_warp_subvolume_rotate(self):
    hy, hx = np.mgrid[-50:50, -50:50]

    # Diamond-oriented box (rhombus).
    image = np.zeros((1, 1, 100, 100), dtype=np.uint8)
    image[0, 0, ...][np.abs(hy) + np.abs(hx) < 25] = 255
    image_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(100, 100, 1))

    # Rotate by 45 deg.
    angle = np.pi / 4
    coord_map = np.zeros((2, 1, 10, 10))
    coord_map[0, 0, :, :] = (np.cos(angle) * hx[::10, ::10] -
                             np.sin(angle) * hy[::10, ::10]) - hx[::10, ::10]
    coord_map[1, 0, :, :] = (np.sin(angle) * hx[::10, ::10] +
                             np.cos(angle) * hy[::10, ::10]) - hy[::10, ::10]
    map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(10, 10, 1))
    edge_len = 10

    out_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(100, 100, 1))
    warped = warp.warp_subvolume(image, image_box, coord_map, map_box, edge_len,
                                 out_box)

    mask = np.zeros((1, 1, 100, 100), dtype=bool)
    mask[0, 0, 33:68, 33:68] = True

    self.assertTrue(np.all(warped[mask] > 128))
    self.assertTrue(np.all(warped[~mask] < 64))

  def test_ndimage_warp_segmentation_translate(self):
    image = np.zeros((100, 100), dtype=np.uint64)
    image[40, 30] = 42
    image[50, 40] = 2**40

    coord_map = np.zeros((2, 25, 25))
    coord_map[0, :, :] = 10
    coord_map[1, :, :] = 17

    warped = warp.ndimage_warp(
        image, coord_map, (4, 5), (100, 100), (0, 0), order=0)
    expected = np.zeros((100, 100))
    expected[23, 20] = 42
    expected[33, 30] = 2**40

    np.testing.assert_array_equal(warped, expected)

  def test_ndimage_warp_3d_translate(self):
    image = np.zeros((10, 100, 100), dtype=np.uint16)
    image[5, 40, 30] = 42
    image[4, 50, 40] = 16

    coord_map = np.zeros((3, 10, 25, 25))
    coord_map[0, :, :] = 10
    coord_map[1, :, :] = 17
    coord_map[2, :, :] = 2

    warped = warp.ndimage_warp(image, coord_map, (1, 4, 5), (50, 50, 5),
                               (2, 2, 2))
    expected = np.zeros((10, 100, 100))
    expected[3, 23, 20] = 42
    expected[2, 33, 30] = 16

    np.testing.assert_array_equal(warped, expected)

  def test_warp_points(self):
    coord_map = np.zeros((2, 10, 3, 3))
    coord_map[0, 0, ...] = 10
    coord_map[1, 1, ...] = 20

    points = np.array([[101, 201, 0], [105, 205, 1]])
    map_box = bounding_box.BoundingBox(start=(10, 20, 0), size=(3, 3, 10))
    stride = 10
    warped = warp.warp_points(points, coord_map, map_box, stride)

    expected = np.array([[111, 201, 0], [105, 225, 1]])
    np.testing.assert_array_equal(warped, expected)

  def _assert_ndimage_warp_multi_matches_reference(
      self,
      images,
      coord_map,
      stride,
      work_size,
      overlap,
      orders,
      image_box,
      map_box,
      out_box,
      parallelism=1,
  ):
    got = warp.ndimage_warp_multi(
        images,
        coord_map,
        stride=stride,
        work_size=work_size,
        overlap=overlap,
        orders=orders,
        image_box=image_box,
        map_box=map_box,
        out_box=out_box,
        parallelism=parallelism,
    )

    def ref_mc(*a, **kw):
      return ndimage.map_coordinates(*a, **kw)

    for got_img, img, order in zip(got, images, orders):
      ref = warp.ndimage_warp(
          img,
          coord_map,
          stride=stride,
          work_size=work_size,
          overlap=overlap,
          order=order,
          map_coordinates=ref_mc,
          image_box=image_box,
          map_box=map_box,
          out_box=out_box,
      )
      np.testing.assert_allclose(got_img, ref, rtol=1e-5, atol=1e-5)

  def test_warp_subvolume_separable_matches_regular_grid_interpolator(self):
    rng = np.random.default_rng(314)
    image = rng.integers(0, 255, size=(1, 3, 96, 104), dtype=np.uint8)
    image_box = bounding_box.BoundingBox(start=(20, 30, 5), size=(104, 96, 3))
    # Map box intentionally smaller than out_box on borders to test linear
    # extrapolation outside [0, size - 1].
    map_box = bounding_box.BoundingBox(start=(3, 4, 5), size=(9, 8, 3))
    stride = 10.0
    coord_map = rng.normal(0.0, 2.5, size=(2, 3, 8, 9)).astype(np.float32)
    out_box = bounding_box.BoundingBox(start=(25, 35, 5), size=(90, 80, 3))

    got = warp.warp_subvolume(
        image,
        image_box,
        coord_map,
        map_box,
        stride,
        out_box,
        interpolation='linear',
        offset=0.25,
    )
    ref = _ref_warp_subvolume_rgi(
        image,
        image_box,
        coord_map,
        map_box,
        stride,
        out_box,
        interpolation=warp.cv.INTER_LINEAR,
        offset=0.25,
    )
    np.testing.assert_array_equal(got, ref)

    # Interior cropped out_box (ky_min > 0, ky_max < hm - 2) matches slice of
    # full warp.
    crop_out_box = bounding_box.BoundingBox(start=(45, 65, 5), size=(40, 25, 3))
    got_crop = warp.warp_subvolume(
        image,
        image_box,
        coord_map,
        map_box,
        stride,
        crop_out_box,
        interpolation='linear',
        offset=0.25,
    )
    np.testing.assert_array_equal(got_crop, ref[:, :, 30:55, 20:60])

  def test_warp_subvolume_length_one_axes_and_empty_out_box(self):
    rng = np.random.default_rng(123)
    image = rng.integers(0, 255, size=(1, 2, 64, 72), dtype=np.uint8)
    image_box = bounding_box.BoundingBox(start=(10, 15, 0), size=(72, 64, 2))
    out_box = bounding_box.BoundingBox(start=(12, 18, 0), size=(50, 44, 2))
    stride = 10.0

    for hm, wm in [(1, 5), (5, 1), (1, 1)]:
      map_box = bounding_box.BoundingBox(start=(2, 3, 0), size=(wm, hm, 2))
      coord_map = rng.normal(0.0, 2.0, size=(2, 2, hm, wm)).astype(np.float32)
      got = warp.warp_subvolume(
          image,
          image_box,
          coord_map,
          map_box,
          stride,
          out_box,
          interpolation='linear',
          offset=0.5,
      )
      ref = _ref_warp_subvolume_rgi(
          image,
          image_box,
          coord_map,
          map_box,
          stride,
          out_box,
          interpolation=warp.cv.INTER_LINEAR,
          offset=0.5,
      )
      np.testing.assert_array_equal(got, ref)

    # Empty out_box returns zero-sized array of original dtype without error.
    coord_map = np.zeros((2, 2, 4, 4), dtype=np.float32)
    map_box = bounding_box.BoundingBox(start=(0, 0, 0), size=(4, 4, 2))
    for empty_size in [(0, 10, 2), (10, 0, 2), (10, 10, 0)]:
      empty_box = bounding_box.BoundingBox(start=(0, 0, 0), size=empty_size)
      for dtype in (np.uint8, np.uint64):
        got = warp.warp_subvolume(
            image.astype(dtype),
            image_box,
            coord_map,
            map_box,
            stride,
            empty_box,
        )
        self.assertEqual(got.dtype, dtype)
        self.assertEqual(got.shape, (1, *empty_size[::-1]))

  def test_ndimage_warp_multi_3d_matches_reference(self):
    rng = np.random.default_rng(2027)
    img3d = rng.uniform(0.0, 100.0, size=(6, 32, 36)).astype(np.float32)
    dts3d = rng.uniform(0.0, 10.0, size=(6, 32, 36)).astype(np.float32)
    cmap3d = rng.normal(0.0, 1.5, size=(3, 4, 9, 10)).astype(np.float32)
    self._assert_ndimage_warp_multi_matches_reference(
        [img3d, dts3d],
        cmap3d,
        stride=(2.0, 4.0, 4.0),
        work_size=(16, 16, 3),
        overlap=(2, 2, 1),
        orders=[2, 1],
        image_box=bounding_box.BoundingBox(start=(10, 20, 2), size=(36, 32, 6)),
        map_box=bounding_box.BoundingBox(start=(1, 2, 1), size=(10, 9, 4)),
        out_box=bounding_box.BoundingBox(start=(12, 22, 2), size=(30, 28, 5)),
        parallelism=2,
    )

  def test_render_tiles_matches_warp_subvolume_reference(self):
    rng = np.random.default_rng(2028)
    t00 = rng.integers(10, 240, size=(120, 120), dtype=np.uint8)
    t10 = rng.integers(10, 240, size=(120, 120), dtype=np.uint8)
    m00 = np.zeros((2, 1, 6, 6), dtype=np.float32)
    m10 = np.zeros((2, 1, 6, 6), dtype=np.float32)
    m10[0, 0] = -20.0
    tmask = {
        (0, 0): np.ones_like(t00, dtype=bool),
        (1, 0): np.ones_like(t10, dtype=bool),
    }
    tmask[(0, 0)][:5, :5] = False
    canvas, cmask, wmap = warp.render_tiles(
        {(0, 0): t00, (1, 0): t10},
        {(0, 0): m00, (1, 0): m10},
        stride=(20, 20),
        margin=10,
        parallelism=2,
        return_warped_tiles=True,
        tile_masks=tmask,
    )
    self.assertEqual(canvas.shape, (120, 240))
    self.assertEqual(cmask.shape, (120, 240))
    self.assertTrue(np.any(cmask))
    self.assertLen(wmap, 2)

    # Verify rendered canvas and mask match full-tile warp_subvolume clipped to
    # the canvas.
    ref_canvas = np.zeros((120, 240), dtype=np.uint8)
    ref_mask = np.zeros((120, 240), dtype=bool)
    img_box_t = bounding_box.BoundingBox(start=(0, 0, 0), size=(120, 120, 1))
    map_box_t = bounding_box.BoundingBox(start=(0, 0, 0), size=(6, 6, 1))
    for (tx, ty), tile_img, cm in [
        ((0, 0), t00, m00),
        ((1, 0), t10, m10),
    ]:
      tg_box = map_utils.outer_box(cm, map_box_t, 20).adjusted_by(
          start=(-1, -1, 0), end=(1, 1, 0)
      )
      inv = map_utils.fill_missing(
          map_utils.invert_map(cm, map_box_t, tg_box, 20), extrapolate=True
      )
      m_arr = np.zeros(tile_img.shape, dtype=np.uint8)
      m_arr[10:-11, 10:-11] = 1
      m_arr[~tmask[(tx, ty)]] = 0
      full_out = bounding_box.BoundingBox(
          start=((tg_box.start[0] + 1) * 20, (tg_box.start[1] + 1) * 20, 0),
          size=(tg_box.size[0] * 20, tg_box.size[1] * 20, 1),
      )
      wi = warp.warp_subvolume(
          tile_img[None, None, ...], img_box_t, inv, tg_box, 20, full_out
      )[0, 0]
      wm = warp.warp_subvolume(
          m_arr[None, None, ...],
          img_box_t,
          inv,
          tg_box,
          20,
          full_out,
          interpolation='linear',
      )[0, 0].astype(bool)
      y_global = np.arange(wi.shape[0]) + 120 * ty + int(full_out.start[1])
      x_global = np.arange(wi.shape[1]) + 120 * tx + int(full_out.start[0])
      ys = (y_global >= 0) & (y_global < 120)
      xs = (x_global >= 0) & (x_global < 240)
      wi_clip = wi[np.ix_(ys, xs)]
      wm_clip = wm[np.ix_(ys, xs)]
      y0, y1 = int(y_global[ys][0]), int(y_global[ys][-1]) + 1
      x0, x1 = int(x_global[xs][0]), int(x_global[xs][-1]) + 1
      ref_mask[y0:y1, x0:x1][wm_clip] = True
      wm_clip &= wi_clip > 0
      ref_canvas[y0:y1, x0:x1][wm_clip] = wi_clip[wm_clip]
    np.testing.assert_array_equal(canvas, ref_canvas)
    np.testing.assert_array_equal(cmask, ref_mask)

  def test_warp_points_matches_regular_grid_interpolator(self):
    rng = np.random.default_rng(2029)
    pts_f = rng.uniform(-15.0, 135.0, size=(200, 3)).astype(np.float32)
    pts_f[:, 2] = rng.integers(5, 8, size=200).astype(np.float32)
    pts_i = np.round(pts_f).astype(np.int64)

    for hm, wm in [(7, 8), (1, 8), (7, 1), (1, 1)]:
      cmap_p = rng.normal(0.0, 3.0, size=(2, 3, hm, wm)).astype(np.float32)
      mbox_p = bounding_box.BoundingBox(start=(1, 2, 5), size=(wm, hm, 3))

      got_pf = warp.warp_points(pts_f, cmap_p, mbox_p, 15.0)
      ref_pf = _ref_warp_points_rgi(pts_f, cmap_p, mbox_p, 15.0)
      np.testing.assert_allclose(got_pf, ref_pf, rtol=1e-5, atol=1e-5)

      got_pi = warp.warp_points(pts_i, cmap_p, mbox_p, 15.0)
      ref_pi = _ref_warp_points_rgi(pts_i, cmap_p, mbox_p, 15.0)
      self.assertEqual(got_pi.dtype, np.int64)
      np.testing.assert_array_equal(got_pi, ref_pi)

  def test_render_tiles_out_box_crop_and_culling(self):
    rng = np.random.default_rng(7)
    tiles = {}
    coord_maps = {}
    for ty in range(3):
      for tx in range(3):
        tiles[(tx, ty)] = rng.integers(10, 240, size=(80, 100), dtype=np.uint8)
        cm = np.zeros((2, 1, 8, 10), dtype=np.float32)
        cm[0, 0] = 2.0 * np.sin(np.arange(10)[None, :] / 4.0)
        cm[1, 0] = -1.5 * np.cos(np.arange(8)[:, None] / 3.0)
        coord_maps[(tx, ty)] = cm

    full_img, full_mask = warp.render_tiles(
        tiles, coord_maps, stride=(10, 10), margin=8
    )
    crop_box = bounding_box.BoundingBox(start=(75, 55, 0), size=(110, 90, 1))
    crop_img, crop_mask, crop_wmap = warp.render_tiles(
        tiles,
        coord_maps,
        stride=(10, 10),
        margin=8,
        out_box=crop_box,
        return_warped_tiles=True,
    )
    np.testing.assert_array_equal(crop_img, full_img[55:145, 75:185])
    np.testing.assert_array_equal(crop_mask, full_mask[55:145, 75:185])
    # Tiles (2, 0), (2, 1), (2, 2), (0, 2), (1, 2) do not intersect crop_box.
    self.assertNotIn((2, 2), crop_wmap)
    for (x0, y0, wtile) in crop_wmap.values():
      self.assertGreaterEqual(x0, 0)
      self.assertGreaterEqual(y0, 0)
      self.assertLessEqual(x0 + wtile.shape[1], 110)
      self.assertLessEqual(y0 + wtile.shape[0], 90)

  def test_render_tiles_dtypes_and_all_nan_map(self):
    rng = np.random.default_rng(11)
    cm_zero = np.zeros((2, 1, 5, 5), dtype=np.float32)

    # uint64 segmentation tile with >32-bit label IDs.
    seg_tile = np.zeros((50, 50), dtype=np.uint64)
    seg_tile[15:35, 15:35] = 2**42 + 7
    seg_out, seg_mask = warp.render_tiles(
        {(0, 0): seg_tile}, {(0, 0): cm_zero}, stride=(10, 10), margin=5
    )
    self.assertEqual(seg_out.dtype, np.uint64)
    self.assertEqual(seg_out[25, 25], 2**42 + 7)
    self.assertTrue(seg_mask[25, 25])
    self.assertFalse(seg_mask[0, 0])

    # uint32 tile within uint16 range vs overflow > 2**16.
    u32_tile = rng.integers(10, 1000, size=(50, 50), dtype=np.uint32)
    u32_out, u32_mask = warp.render_tiles(
        {(0, 0): u32_tile}, {(0, 0): cm_zero}, stride=(10, 10), margin=5
    )
    u16_out, u16_mask = warp.render_tiles(
        {(0, 0): u32_tile.astype(np.uint16)},
        {(0, 0): cm_zero},
        stride=(10, 10),
        margin=5,
    )
    self.assertEqual(u32_out.dtype, np.uint32)
    np.testing.assert_array_equal(u32_out, u16_out.astype(np.uint32))
    np.testing.assert_array_equal(u32_mask, u16_mask)

    u32_overflow = u32_tile.copy()
    u32_overflow[20, 20] = 2**16
    with self.assertRaises(ValueError):
      warp.render_tiles(
          {(0, 0): u32_overflow}, {(0, 0): cm_zero}, stride=(10, 10), margin=5
      )

    # Sparse coordinate map (1 valid node) whose inverted_map is all-NaN
    # produces a zero image and empty mask without failing.
    cm_nan = np.full((2, 1, 5, 5), np.nan, dtype=np.float32)
    cm_nan[:, 0, 0, 0] = 0.0
    for dtype in (np.uint8, np.uint64):
      nan_tile = np.full((50, 50), 123, dtype=dtype)
      nan_out, nan_mask, nan_wmap = warp.render_tiles(
          {(0, 0): nan_tile},
          {(0, 0): cm_nan},
          stride=(10, 10),
          margin=5,
          return_warped_tiles=True,
      )
      np.testing.assert_array_equal(nan_out, np.zeros((50, 50), dtype=dtype))
      np.testing.assert_array_equal(nan_mask, np.zeros((50, 50), dtype=bool))
      self.assertIn((0, 0), nan_wmap)
      self.assertEqual(nan_wmap[(0, 0)][2].dtype, dtype)

  def test_ndimage_warp_2d_with_map_box(self):
    rng = np.random.default_rng(42)
    img2d = rng.uniform(0.0, 100.0, size=(40, 48)).astype(np.float32)
    dts2d = rng.uniform(0.0, 10.0, size=(40, 48)).astype(np.float32)
    cmap2d = rng.normal(0.0, 1.5, size=(2, 10, 12)).astype(np.float32)
    self._assert_ndimage_warp_multi_matches_reference(
        [img2d, dts2d],
        cmap2d,
        stride=(4.0, 4.0),
        work_size=(16, 16),
        overlap=(2, 2),
        orders=[1, 1],
        image_box=bounding_box.BoundingBox(start=(8, 12, 0), size=(48, 40, 1)),
        map_box=bounding_box.BoundingBox(start=(2, 3, 0), size=(12, 10, 1)),
        out_box=bounding_box.BoundingBox(start=(10, 14, 0), size=(36, 32, 1)),
    )

  def test_separable_linear_upsample_edge_cases(self):
    # 1. m_len == 1 with NaN and finite entries + OOB coordinates.
    eval_1 = np.array([[5.0, 7.0, np.nan]], dtype=np.float64)
    q0 = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    q1 = np.array([-1.0, 0.0, 0.5, 1.5, 2.0, 3.0], dtype=np.float64)
    got_1 = warp._separable_linear_upsample(eval_1, [q0, q1])
    grid0, grid1 = np.meshgrid(q0, q1, indexing='ij')
    ref_1 = ndimage.map_coordinates(
        eval_1, [grid0, grid1], order=1, mode='constant', cval=0.0
    )
    np.testing.assert_allclose(got_1, ref_1, equal_nan=True)
    # OOB rows/cols must be exact 0.0 even next to NaN border entries.
    np.testing.assert_array_equal(got_1[0, :], 0.0)
    np.testing.assert_array_equal(got_1[2, :], 0.0)
    np.testing.assert_array_equal(got_1[:, 0], 0.0)
    np.testing.assert_array_equal(got_1[:, -1], 0.0)

    # 2. m_len >= 2 with NaN/inf at edges and OOB coordinates.
    eval_nan = np.array(
        [
            [np.nan, 1.0, 2.0, np.inf],
            [3.0, 4.0, 5.0, 6.0],
            [-np.inf, 7.0, 8.0, np.nan],
        ],
        dtype=np.float64,
    )
    qa = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 2.5], dtype=np.float64)
    qb = np.array([-0.2, 0.0, 1.25, 2.0, 3.0, 4.0], dtype=np.float64)
    got_nan = warp._separable_linear_upsample(eval_nan, [qa, qb])
    ga, gb = np.meshgrid(qa, qb, indexing='ij')
    ref_nan = ndimage.map_coordinates(
        eval_nan, [ga, gb], order=1, mode='constant', cval=0.0
    )
    np.testing.assert_allclose(got_nan, ref_nan, equal_nan=True)
    np.testing.assert_array_equal(got_nan[0, :], 0.0)
    np.testing.assert_array_equal(got_nan[-1, :], 0.0)
    np.testing.assert_array_equal(got_nan[:, 0], 0.0)
    np.testing.assert_array_equal(got_nan[:, -1], 0.0)

    # 3. All-OOB along one axis returns exact zeros without NaN contamination.
    got_oob = warp._separable_linear_upsample(
        eval_nan, [np.array([5.0, 6.0]), np.array([0.0, 1.0])]
    )
    np.testing.assert_array_equal(got_oob, np.zeros((2, 2), dtype=np.float64))

    # 4. Large map with small sub-window matches map_coordinates exactly.
    rng = np.random.default_rng(99)
    large_map = rng.normal(size=(8, 120, 120))
    qz = np.linspace(2.0, 3.5, 4)
    qy = np.linspace(50.2, 52.8, 12)
    qx = np.linspace(80.1, 82.9, 10)
    got_sub = warp._separable_linear_upsample(large_map, [qz, qy, qx])
    gz, gy, gx = np.meshgrid(qz, qy, qx, indexing='ij')
    ref_sub = ndimage.map_coordinates(
        large_map, [gz, gy, gx], order=1, mode='constant', cval=0.0
    )
    np.testing.assert_allclose(got_sub, ref_sub, rtol=1e-12, atol=1e-12)

  def test_warp_points_fractional_z_truncation(self):
    coord_map = np.zeros((2, 3, 4, 4), dtype=np.float32)
    coord_map[0, 0, :, :] = 10.0
    coord_map[0, 1, :, :] = 20.0
    coord_map[0, 2, :, :] = 30.0
    map_box = bounding_box.BoundingBox(start=(0, 0, 5), size=(4, 4, 3))

    # Include fractional Z >= end - 0.5 (7.9) which must truncate to z_rel=2
    # rather than rounding to 3 and raising IndexError.
    pts = np.array(
        [[10.0, 10.0, 5.7], [10.0, 10.0, 6.6], [10.0, 10.0, 7.9]],
        dtype=np.float32,
    )
    got = warp.warp_points(pts, coord_map, map_box, stride=10.0)
    expected = np.array(
        [[20.0, 10.0, 5.7], [30.0, 10.0, 6.6], [40.0, 10.0, 7.9]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    # Matches the hm < 2 fallback path on the same fractional Z coordinates
    # evaluated at the shared (0, 0) map node.
    pts_origin = np.array(
        [[0.0, 0.0, 5.7], [0.0, 0.0, 6.6], [0.0, 0.0, 7.9]],
        dtype=np.float32,
    )
    got_origin = warp.warp_points(pts_origin, coord_map, map_box, stride=10.0)
    coord_map_1d = coord_map[:, :, :1, :1]
    map_box_1d = bounding_box.BoundingBox(start=(0, 0, 5), size=(1, 1, 3))
    got_fallback = warp.warp_points(
        pts_origin, coord_map_1d, map_box_1d, stride=10.0
    )
    np.testing.assert_allclose(got_origin, got_fallback, rtol=1e-6, atol=1e-6)

  def test_stitch_and_render_3d_tiles_dts_cache(self):
    tile_map_3x3 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    proc_no_margin = warp_processor.StitchAndRender3dTiles(
        tile_map=tile_map_3x3,
        tile_mesh_path='',
        tile_pattern_path='',
        stride=(1, 4, 4),
        margin=0,
    )
    ref_dts = proc_no_margin._get_dts((2, 20, 24), 0, 0)
    self.assertFalse(ref_dts.flags.writeable)
    with self.assertRaises(ValueError):
      ref_dts[0, 0] = 0.0
    for ty in range(3):
      for tx in range(3):
        self.assertIs(proc_no_margin._get_dts((2, 20, 24), tx, ty), ref_dts)
    self.assertLen(proc_no_margin._dts_cache, 1)

    proc_margin = warp_processor.StitchAndRender3dTiles(
        tile_map=tile_map_3x3,
        tile_mesh_path='',
        tile_pattern_path='',
        stride=(1, 4, 4),
        margin=3,
    )
    for ty in range(3):
      for tx in range(3):
        dts = proc_margin._get_dts((2, 20, 24), tx, ty)
        self.assertFalse(dts.flags.writeable)
        self.assertIs(proc_margin._get_dts((2, 20, 24), tx, ty), dts)
    self.assertLen(proc_margin._dts_cache, 9)

  def test_stitch_and_render_3d_tiles_process(self):
    # End-to-end process() check on a small in-memory 2x1 tile grid.
    tile0 = np.full((2, 16, 16), 100, dtype=np.uint8)
    tile1 = np.full((2, 16, 16), 200, dtype=np.uint8)
    vols = {0: tile0, 1: tile1}

    class _InMemoryStitcher(warp_processor.StitchAndRender3dTiles):

      def _open_tile_volume(self, tile_id: int):
        return vols[tile_id]

    tile_meshes = np.zeros((3, 2, 2, 4, 4), dtype=np.float32)
    # Overlap tile 1 leftward by 4 pixels into tile 0.
    tile_meshes[0, 1, ...] = -4.0
    cls = warp_processor.StitchAndRender3dTiles
    with (
        mock.patch.object(cls, '_tile_meshes', tile_meshes),
        mock.patch.object(cls, '_tile_idx_to_xy', {0: (0, 0), 1: (1, 0)}),
        mock.patch.object(cls, '_tile_boxes', {}),
        mock.patch.object(cls, '_inverted_meshes', {}),
    ):
      proc = _InMemoryStitcher(
          tile_map=[[0, 1]],
          tile_mesh_path='',
          tile_pattern_path='',
          stride=(1, 4, 4),
          margin=0,
          work_size=(16, 16, 2),
          parallelism=2,
      )
      proc._collect_tile_boxes(tile0.shape)
      box = bounding_box.BoundingBox(start=(0, 0, 0), size=(28, 16, 2))
      proc.set_effective_subvol_and_overlap(box.size, (0, 0, 0))
      inp = subvolume.Subvolume(
          np.zeros((1, 2, 16, 28), dtype=np.uint8), box
      )
      out = proc.process(inp)
      self.assertIsInstance(out, subvolume.Subvolume)
      # Interior of tile 0 -> ~100, interior of tile 1 -> ~200, overlap blended.
      self.assertEqual(int(out.data[0, 0, 8, 4]), 100)
      self.assertEqual(int(out.data[0, 0, 8, 22]), 200)
      overlap_val = int(out.data[0, 0, 8, 13])
      self.assertGreater(overlap_val, 100)
      self.assertLess(overlap_val, 200)


if __name__ == '__main__':
  absltest.main()
