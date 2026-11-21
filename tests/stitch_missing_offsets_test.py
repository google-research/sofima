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
"""Missing coarse-offset sentinels must not create spring forces."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from sofima import mesh
from sofima import stitch_rigid


def reference_forces(x, cx, cy):
  """Explicit pairwise force balance, excluding unavailable link components."""
  forces = np.zeros_like(x)
  for channel in range(x.shape[0]):
    for z in range(x.shape[1]):
      for y in range(x.shape[2]):
        for col in range(x.shape[3]):
          for dy, dx, offsets in [(0, 1, cx), (1, 0, cy)]:
            if y + dy >= x.shape[2] or col + dx >= x.shape[3]:
              continue
            desired = offsets[channel, z, y, col]
            if not np.isfinite(desired):
              continue
            delta = (
                x[channel, z, y + dy, col + dx]
                - x[channel, z, y, col]
                - desired
            )
            forces[channel, z, y, col] += delta
            forces[channel, z, y + dy, col + dx] -= delta
  return forces


class MissingOffsetsTest(parameterized.TestCase):

  @parameterized.parameters(
      (2, np.inf),
      (2, -np.inf),
      (2, np.nan),
      (3, np.inf),
      (3, -np.inf),
      (3, np.nan),
  )
  def test_unavailable_links_exert_no_force(self, dim, missing):
    fn = (
        stitch_rigid.elastic_tile_mesh
        if dim == 2
        else stitch_rigid.elastic_tile_mesh_3d
    )
    x = jnp.arange(dim * 6, dtype=jnp.float32).reshape(dim, 1, 2, 3)
    offsets = jnp.full_like(x, missing)
    for evaluation in (fn, jax.jit(fn)):
      actual = evaluation(x, offsets, offsets)
      np.testing.assert_array_equal(actual, jnp.zeros_like(x))

  @parameterized.parameters(2, 3)
  def test_mixed_links_match_independent_pairwise_force_balance(self, dim):
    fn = (
        stitch_rigid.elastic_tile_mesh
        if dim == 2
        else stitch_rigid.elastic_tile_mesh_3d
    )
    rng = np.random.default_rng(9)
    x = rng.normal(size=(dim, 2, 3, 4)).astype(np.float32)
    cx = rng.normal(size=x.shape).astype(np.float32)
    cy = rng.normal(size=x.shape).astype(np.float32)
    cx[:, :, 0, 0] = np.inf
    cy[:, :, 0, 1] = -np.inf
    cx[:, :, 1, 1] = np.nan
    expected = reference_forces(x, cx, cy)
    for evaluation in (fn, jax.jit(fn)):
      actual = evaluation(jnp.asarray(x), jnp.asarray(cx), jnp.asarray(cy))
      np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
      np.testing.assert_allclose(np.sum(actual, axis=(1, 2, 3)), 0, atol=3e-6)

  @parameterized.parameters(2, 3)
  def test_finite_zero_offsets_still_produce_real_forces(self, dim):
    fn = (
        stitch_rigid.elastic_tile_mesh
        if dim == 2
        else stitch_rigid.elastic_tile_mesh_3d
    )
    x = np.arange(dim * 6, dtype=np.float32).reshape(dim, 1, 2, 3)
    cx = np.zeros_like(x)
    np.testing.assert_allclose(
        fn(jnp.asarray(x), jnp.asarray(cx), jnp.asarray(cx)),
        reference_forces(x, cx, cx),
        rtol=0,
        atol=0,
    )

  @parameterized.parameters(2, 3)
  def test_missing_links_have_zero_coordinate_derivatives(self, dim):
    fn = (
        stitch_rigid.elastic_tile_mesh
        if dim == 2
        else stitch_rigid.elastic_tile_mesh_3d
    )
    x = jnp.ones((dim, 1, 2, 3))
    offsets = jnp.full_like(x, jnp.inf)
    values, tangent = jax.jvp(lambda p: fn(p, offsets, offsets), (x,), (x,))
    np.testing.assert_array_equal(values, jnp.zeros_like(x))
    np.testing.assert_array_equal(tangent, jnp.zeros_like(x))

  def test_coarse_optimizer_stays_finite_with_unavailable_connections(self):
    offsets = np.full((2, 1, 2, 2), np.inf, dtype=np.float32)
    cfg = mesh.IntegrationConfig(
        dt=0.001,
        gamma=0.1,
        k0=0.0,
        k=0.1,
        stride=(1, 1),
        num_iters=2,
        max_iters=2,
        stop_v_max=0.001,
    )
    actual = stitch_rigid.optimize_coarse_mesh(offsets, offsets, cfg)
    np.testing.assert_array_equal(actual, np.zeros_like(offsets))


if __name__ == '__main__':
  absltest.main()
