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
"""Tests for flow field decorators."""

from absl.testing import absltest
import numpy as np
import sofima.decorators.flow as decorators
import tensorstore as ts


class DecoratorsTest(absltest.TestCase):

  def test_clean_flow_filter(self):
    flow_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (5, 3, 3, 3),
        },
        'create': True,
        'delete_existing': True,
    }
    f = ts.open(flow_spec).result()
    data = np.array(np.random.uniform(size=f.schema.shape), dtype='float32')
    f[...] = data
    filter_args = {
        'min_peak_sharpness': 1.6,
        'min_peak_ratio': 1.4,
        'max_magnitude': 20,
        'max_deviation': 2,
    }
    dec = decorators.CleanFlowFilter(
        min_chunksize=f.shape, **filter_args)
    vc = dec.decorate(f)
    res = vc[...].read().result()
    np.testing.assert_equal(
        res, decorators._clean_flow(data, **filter_args))

  def test_mesh_relax_flow_filter(self):
    flow_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (3, 3, 3, 3),
        },
        'create': True,
        'delete_existing': True,
    }
    f = ts.open(flow_spec).result()
    data = np.array(np.random.uniform(size=f.schema.shape), dtype='float32')
    f[...] = data
    filter_args = {
        'k0': 0.1,
        'k': 0.1,
        'dt': 0.001,
        'gamma': 0.0,
        'stride': (1, 1, 1),  # XYZ
        'num_iters': 1000,
        'max_iters': 50_000,
        'stop_v_max': 0.001,
        'dt_max': 1000,
    }
    dec = decorators.MeshRelaxFlowFilter(
        min_chunksize=f.shape, **filter_args)
    vc = dec.decorate(f)
    res = vc[...].read().result()
    np.testing.assert_equal(
        res, decorators._mesh_relax_flow(data, **filter_args))

  def test_optim_flow(self):
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
    img_ts[...].write(np.zeros(img_spec['metadata']['dimensions'],
                               dtype=np.float32)).result()

    img_spec['open'] = True
    img_spec['create'] = False
    img_spec['delete_existing'] = False

    vc = decorators.OptimFlow(
        fixed_spec=img_spec,
        image_dims=('x', 'y',),
    )
    dec = vc.decorate(img_ts)
    assert dec.domain.labels == ('fc', 'fz', 'fy', 'fx')

  def test_reconcile_flow_filter(self):
    flow_spec = {
        'driver': 'n5',
        'kvstore': {
            'driver': 'file',
            'path': self.create_tempdir().full_path,
        },
        'metadata': {
            'dataType': 'float32',
            'dimensions': (3, 3, 3, 3),
        },
        'create': True,
        'delete_existing': True,
    }
    f = ts.open(flow_spec).result()
    data = np.array(np.random.uniform(size=f.schema.shape), dtype='float32')
    f[...] = data
    filter_args = {
        'max_gradient': 2.0,
        'max_deviation': 2,
        'min_patch_size': 20,
    }
    dec = decorators.ReconcileFlowFilter(
        min_chunksize=f.shape, **filter_args)
    vc = dec.decorate(f)
    res = vc[...].read().result()
    np.testing.assert_equal(
        res, decorators._reconcile_flow(data, **filter_args))


if __name__ == '__main__':
  absltest.main()
