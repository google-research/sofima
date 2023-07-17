# coding=utf-8
# Copyright 2023 The Google Research Authors.
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
"""Tests for image warping and rendering decorators."""

from absl.testing import absltest
import numpy as np
import sofima.decorators.warp as decorators
import tensorstore as ts


class DecoratorsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._data = ts.open({
        'driver': 'n5',
        'kvstore': {
            'driver': 'memory',
        },
        'metadata': {
            'dataType': 'float64',
            'dimensions': (10, 10, 10),
            'blockSize': (1, 1, 1),
            'axes': ('x', 'y', 'z'),
        },
        'create': True,
        'delete_existing': True,
    }).result()
    rng = np.random.default_rng(seed=42)
    self._data[...] = np.array(
        rng.uniform(size=self._data.schema.shape), dtype=np.float64)

  def test_warp_affine(self):
    affine_transform = np.array([
        [1., 0., 10.],
        [0., 1., 0.]], dtype=np.float32)

    transform_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (2, 3),
            'axes': ('r', 'c'),
        },
        'create': True,
        'delete_existing': True,
    }
    transform_ts = ts.open(transform_spec).result()
    transform_ts[...].write(affine_transform).result()

    transform_spec['create'] = False
    transform_spec['delete_existing'] = False
    transform_spec['open'] = True

    for implementation in ['scipy', 'opencv']:
      dec = decorators.WarpAffine(
          transform_spec=transform_spec, implementation=implementation)
      vc = dec.decorate(self._data)

      np.testing.assert_equal(
          vc[..., 0].read().result(),
          decorators._warp_affine(
              np.array(self._data[..., 0]), affine_transform,
              implementation=implementation))

  def test_warp_coord_map(self):
    transform_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (3, 10, 10, 10),
            'axes': ('fc', 'fz', 'fy', 'fx'),
        },
        'create': True,
        'delete_existing': True,
    }
    transform_ts = ts.open(transform_spec).result()
    coord_map = np.zeros(transform_spec['metadata']['dimensions'],
                         dtype=np.float32)
    transform_ts[...].write(coord_map).result()

    transform_spec['create'] = False
    transform_spec['delete_existing'] = False
    transform_spec['open'] = True

    warp_args = {
        'work_size': (10, 10, 10),  # XYZ
        'parallelism': 1,
        'stride': (1, 1, 1),  # ZYX
        'order': 1,
        'overlap': (0, 0, 0)
    }
    dec = decorators.WarpCoordMap(
        coord_map_spec=transform_spec, **warp_args)
    vc = dec.decorate(self._data)

    np.testing.assert_equal(
        vc[...].read().result(),
        decorators._warp_coord_map(
            np.array(self._data[...]), coord_map, **warp_args))


if __name__ == '__main__':
  absltest.main()
