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

from absl.testing import absltest
from connectomics.common import bounding_box
import numpy as np

from sofima import warp


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


if __name__ == '__main__':
  absltest.main()
