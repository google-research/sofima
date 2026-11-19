# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Tests for SOFIMA mesh processor."""

from absl.testing import absltest
import numpy as np
from sofima import mesh as mesh_lib
from sofima.processor import mesh as mesh_proc


def _make_config(
    output_dir: str, options: mesh_proc.MeshOptions | None
) -> mesh_proc.RelaxMesh.Config:
  icfg = mesh_lib.IntegrationConfig(
      dt=0.001,
      gamma=0.0,
      k0=0.01,
      k=0.1,
      stride=(40.0, 40.0),
      num_iters=100,
      max_iters=1000,
      stop_v_max=0.005,
  )
  return mesh_proc.RelaxMesh.Config(
      output_dir=output_dir,
      integration_config=icfg,
      mesh=None,
      flows=[],
      sections_to_skip=[],
      ranges_to_skip=[],
      mask=None,
      block_starts=[0],
      block_ends=[10],
      backward=False,
      mesh_min_frac=0.5,
      mesh_max_frac=2.0,
      coming_in=[],
      options=options,
  )


class RelaxMeshProcessorTest(absltest.TestCase):

  def test_maybe_update_init_state_accepts_none_options(self):
    proc = mesh_proc.RelaxMesh(
        _make_config(self.create_tempdir().full_path, options=None)
    )
    x = np.ones((2, 1, 4, 4), dtype=np.float32) * 3.0
    prev = np.full((2, 1, 4, 4), 7.0, dtype=np.float32)

    out = proc.maybe_update_init_state(x.copy(), prev, None)
    np.testing.assert_array_equal(out, x)

  def test_maybe_update_init_state_prev_median(self):
    opts = mesh_proc.MeshOptions(init_state=mesh_proc.MeshInitState.PREV_MEDIAN)
    proc = mesh_proc.RelaxMesh(
        _make_config(self.create_tempdir().full_path, options=opts)
    )
    x = np.zeros((2, 1, 2, 2), dtype=np.float32)
    prev = np.array(
        [[[[1.0, 3.0], [5.0, np.nan]]], [[[-2.0, -4.0], [-6.0, np.nan]]]],
        dtype=np.float32,
    )

    out = proc.maybe_update_init_state(x, prev, opts)
    np.testing.assert_allclose(out[0], 3.0)
    np.testing.assert_allclose(out[1], -4.0)


if __name__ == '__main__':
  absltest.main()
