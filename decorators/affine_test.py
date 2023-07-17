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
"""Tests for affine transform decorators."""

from absl.testing import absltest
import numpy as np
import sofima.decorators.affine as decorators
import tensorstore as ts


class DecoratorsTest(absltest.TestCase):

  def test_optim_affine_transform_sectionwise(self):
    img_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (3, 3,),
            'axes': ('x', 'y',),
        },
        'create': True,
        'delete_existing': True,
    }
    img_ts = ts.open(img_spec).result()
    img_ts[...].write(np.array([[0.1, 0.2, 0.3],
                                [0.1, 0.2, 0.3],
                                [0.1, 0.2, 0.3]], dtype=np.float32)).result()

    img_spec['open'] = True
    img_spec['create'] = False
    img_spec['delete_existing'] = False

    vc = decorators.OptimAffineTransformSectionwise(
        fixed_spec=img_spec,
        image_dims=('x', 'y',),
    )
    dec = vc.decorate(img_ts)
    assert dec.domain.labels == ('r', 'c')

    res = dec[...].read().result()
    expected_res = np.zeros_like(res)
    expected_res[0, 0] = 1.
    expected_res[1, 1] = 1.
    np.testing.assert_almost_equal(res, expected_res, decimal=5)

  def test_optim_translation_transform(self):
    img_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (3, 3,),
            'axes': ('x', 'y',),
        },
        'create': True,
        'delete_existing': True,
    }
    img_ts = ts.open(img_spec).result()
    img_ts[...].write(np.array([[0.1, 0.2, 0.3],
                                [0.1, 0.2, 0.3],
                                [0.1, 0.2, 0.3]], dtype=np.float32)).result()

    img_spec['open'] = True
    img_spec['create'] = False
    img_spec['delete_existing'] = False

    vc = decorators.OptimTranslationTransform(
        fixed_spec=img_spec,
        image_dims=('x', 'y',),
    )
    dec = vc.decorate(img_ts)
    assert dec.domain.labels == ('r', 'c')

    res = dec[...].read().result()
    expected_res = np.zeros_like(res)
    expected_res[0, 0] = 1.
    expected_res[1, 1] = 1.
    expected_res[0, 2] = 0.
    expected_res[0, 2] = 0.
    np.testing.assert_almost_equal(res, expected_res, decimal=5)


if __name__ == '__main__':
  absltest.main()
