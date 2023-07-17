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
"""Tests for coordinate maps decorators."""

from absl.testing import absltest
import numpy as np
import sofima.decorators.maps as decorators
import tensorstore as ts


class DecoratorsTest(absltest.TestCase):

  def test_compose_coord_maps(self):
    size = (3, 10, 10, 10)
    coord_map = np.ones(size, dtype=np.float32)

    map1_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': size,
            'axes': ('fc', 'fz', 'fy', 'fx'),
        },
        'create': True,
        'delete_existing': True,
    }
    map1_ts = ts.open(map1_spec).result()
    map1_ts[...].write(coord_map).result()
    map1_spec['create'] = False
    map1_spec['delete_existing'] = False
    map1_spec['open'] = True

    map2_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': size,
            'axes': ('fc', 'fz', 'fy', 'fx'),
        },
        'create': True,
        'delete_existing': True,
    }
    map2_ts = ts.open(map2_spec).result()
    map2_ts[...].write(coord_map).result()
    map2_spec['create'] = False
    map2_spec['delete_existing'] = False
    map2_spec['open'] = True

    compose_args = {
        'start1': [0., 0., 0.],
        'start2': [0., 0., 0.],
        'stride1': [1, 1, 1],
        'stride2': [1, 1, 1]
    }
    dec = decorators.ComposeCoordMaps(
        coord_map_spec=map2_spec, **compose_args)
    vc = dec.decorate(map1_ts)

    np.testing.assert_equal(vc[...].read().result()[:, 2:-2, 2:-2, 2:-2],
                            2. * coord_map[:, 2:-2, 2:-2, 2:-2])

  def test_make_affine_coord_map(self):
    affine_transform = np.array([
        [1., 0., 0., 1.],
        [0., 1., 0., 2.],
        [0., 0., 1., 3.]], dtype=np.float32)

    transform_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (3, 4),
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

    dec = decorators.MakeAffineCoordMap(size=(10, 10, 10))
    vc = dec.decorate(transform_ts)
    res = vc[...].read().result()

    np.testing.assert_equal(res[0, ...], 1. * np.ones((10, 10, 10)))
    np.testing.assert_equal(res[1, ...], 2. * np.ones((10, 10, 10)))
    np.testing.assert_equal(res[2, ...], 3. * np.ones((10, 10, 10)))


if __name__ == '__main__':
  absltest.main()
