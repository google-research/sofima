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

"""Tests for mesh."""

from absl.testing import absltest
import numpy as np
from sofima import mesh


class MeshTest(absltest.TestCase):

  def test_relaxation_fire(self):
    x = np.zeros((2, 1, 50, 50))
    x[0, 0, 20:30, 10] = 3
    x[0, 0, 20:30, 40] = -4
    x[1, 0, 30, 10:20] = 2
    config = mesh.IntegrationConfig(
        dt=0.01,
        gamma=0.0,
        k0=0.1,
        k=0.1,
        stride=(10, 10),
        num_iters=100,
        max_iters=10000,
        stop_v_max=0.001,
        fire=True,
    )
    new_x, _, _ = mesh.relax_mesh(x, np.zeros_like(x), config)
    new_x = np.array(new_x)

    np.testing.assert_array_almost_equal(new_x, np.zeros_like(x), decimal=3)

  def test_relaxation_damped(self):
    x = np.zeros((2, 1, 50, 50))
    x[0, 0, 20:30, 10] = 3
    x[0, 0, 20:30, 40] = -4
    x[1, 0, 30, 10:20] = 2
    config = mesh.IntegrationConfig(
        dt=0.01,
        gamma=0.9 * np.sqrt(4 * 0.1),  # 90% of critical damping
        k0=0.1,
        k=0.1,
        stride=(10, 10),
        num_iters=100,
        max_iters=10000,
        stop_v_max=0.001,
        fire=False,
    )
    new_x, _, _ = mesh.relax_mesh(x, np.zeros_like(x), config)
    new_x = np.array(new_x)

    np.testing.assert_array_almost_equal(new_x, np.zeros_like(x), decimal=3)

  def test_inplane_equilibrium(self):
    x = np.zeros((2, 1, 10, 10))
    f = np.array(mesh.inplane_force(x, k=1.0, stride=(40.0, 40.0)))
    np.testing.assert_array_equal(x, f)

  def test_3d_equilibrium(self):
    x = np.zeros((3, 10, 10, 10))
    f = np.array(mesh.elastic_mesh_3d(x, k=1.0, stride=40.0))
    np.testing.assert_array_equal(x, f)

    # Add a batch dimension.
    x = np.zeros((3, 5, 10, 10, 10))
    f = np.array(mesh.elastic_mesh_3d(x, k=1.0, stride=40.0))
    np.testing.assert_array_equal(x, f)

  def test_force(self):
    x = np.zeros((2, 1, 10, 10))
    dx = 4
    dy = -3
    x[0, 0, 5, 5] = dx
    x[1, 0, 5, 5] = dy

    k = 0.1
    l0 = 10.0
    f = np.array(mesh.inplane_force(x, k=k, stride=(l0, 10)))

    expected = np.zeros((2, 1, 10, 10))

    # Force on the left neighbor of the perturbed node.
    l = np.sqrt((l0 + dx) ** 2 + dy**2)
    expected[0, 0, 5, 4] = k * (l - l0) * (l0 + dx) / l
    expected[1, 0, 5, 4] = k * (l - l0) * dy / l
    np.testing.assert_allclose(expected[:, 0, 5, 4], f[:, 0, 5, 4], rtol=1e-6)

    # Force on the top neighbor of the perturbed node.
    l = np.sqrt(dx**2 + (l0 + dy) ** 2)
    expected[0, 0, 4, 5] = k * (l - l0) * dx / l
    expected[1, 0, 4, 5] = k * (l - l0) * (l0 + dy) / l
    np.testing.assert_allclose(expected[:, 0, 4, 5], f[:, 0, 4, 5], rtol=1e-6)

    # Force on the bottom-right neighbor of the perturbed node.
    l = np.sqrt((l0 - dx) ** 2 + (l0 - dy) ** 2)
    l2 = l0 * np.sqrt(2.0)
    k2 = k / np.sqrt(2.0)
    expected[0, 0, 6, 6] = -k2 * (l - l2) * (l0 - dx) / l
    expected[1, 0, 6, 6] = -k2 * (l - l2) * (l0 - dy) / l
    np.testing.assert_allclose(expected[:, 0, 6, 6], f[:, 0, 6, 6], rtol=1e-5)

    # Force on the bottom-left neighbor of the perturbed node.
    l = np.sqrt((l0 + dx) ** 2 + (l0 - dy) ** 2)
    l2 = l0 * np.sqrt(2.0)
    expected[0, 0, 6, 4] = k2 * (l - l2) * (l0 + dx) / l
    expected[1, 0, 6, 4] = -k2 * (l - l2) * (l0 - dy) / l
    np.testing.assert_allclose(expected[:, 0, 6, 4], f[:, 0, 6, 4], rtol=1e-5)

  def test_2d_3d_consistency(self):
    planar_directions = (  # xyz
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (-1, 1, 0),
    )

    rng = np.random.default_rng(42)
    x = rng.random((3, 1, 50, 50))
    x[2, ...] = 0.0  # ensure nodes are planar

    f_2d = mesh.inplane_force(x[:2], 0.01, (40.0, 40.0), False)
    f_3d = mesh.elastic_mesh_3d(
        x, 0.01, (40.0, 40.0, 14.0), False, links=planar_directions
    )
    np.testing.assert_allclose(f_2d[:2], f_3d[:2], atol=1e-5)

    f_2d = mesh.inplane_force(x[:2], 0.01, (40.0, 40.0), True)
    f_3d = mesh.elastic_mesh_3d(
        x, 0.01, (40.0, 40.0, 14.0), True, links=planar_directions
    )
    np.testing.assert_allclose(f_2d[:2], f_3d[:2], atol=1e-5)


if __name__ == '__main__':
  absltest.main()
