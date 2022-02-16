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

"""Code for simulating spring mesh systems.

Minimizes the energy of a rectangular grid of Hookean springs with nearest-
and next-nearest neighbor connections using numerical integration of the
corresponding ODE system. Energy minimization is achieved by physically
simulating dissipation with damping, or indirectly in the integrator when
the FIRE method is used.

Positions are always stored in relative format, i.e. the (i, j)-th node
of the grid with stride Δ having value (Δx, Δy) represents the physical
position of (i * Δ + Δx, j * Δ + Δy).
"""
import collections
import dataclasses
import functools
from typing import List, Optional, Sequence, Tuple, Union

from absl import logging

import dataclasses_json

import jax
import jax.numpy as jnp
import numpy as np


# NOTE: This is likely a good candidate for acceleration with a custom CUDA
# kernel on GPUs.
def inplane_force(x: jnp.ndarray,
                  k: float,
                  stride: float,
                  prefer_orig_order=False) -> jnp.ndarray:
  """Computes in-plane forces on the nodes of a spring mesh.

  Args:
    x: [2, z, y, x] array of mesh node positions, in relative format
    k: spring constant
    stride: XY stride of the spring mesh grid
    prefer_orig_order: whether to change the force formulation so that the
      original relative spatial ordering of the nodes is energetically preferred

  Returns:
    [2, z, y, x] array of forces
  """
  l0 = stride
  l0_diag = jnp.sqrt(2.0) * l0

  def _xy_vec(x, y):
    return jnp.array([x, y]).reshape([2, 1, 1, 1])

  # Normal Hookean springs have a force discontinuity at length = 0. When the
  # springs are arranged in a mesh, and the order of neighboring nodes flips
  # during mesh simulation, the new configuration (m2) is energetically
  # preferred:
  #
  #  force
  #   ^        \        \
  #   |          \     .  \
  #   +-----------m2---*---m1-----> x
  #   |             \  .     \
  #   |               \        \
  #
  # (the diagram illustrates the force on a mesh node adjacent to a reference
  #  node indicated by *)
  #
  # This is undesirable as it favors the formation of mesh folds in the presence
  # of external forces strong enough to overcome the spring resistance.
  #
  # To avoid this, we introduce a vector field factor for l0 in the formulas
  # below, which causes the original mesh node order to be favored, and
  # modifies the force so that only a single minimum at m1 exists:
  #
  #  force
  #   ^               \
  #   |                 \
  #   |                   \
  #   +----------------*---m1-----> x
  #   |                      \
  #   |                        \
  #
  # In changing the formulation in this way we sacrifice the ability of the mesh
  # to rotate (which would flip the node order), which is fine since this code
  # is expected to be used on data with prealigned sections with no significant
  # relative rotation.
  #
  # As of Sep 2021, using the fold-preventing force formulation can cause up to
  # a 50% performance penalty on a P100/V100 GPU.
  #
  # - springs
  dx = x[..., 1:] - x[..., :-1] + _xy_vec(l0, 0)
  l = jnp.linalg.norm(dx, axis=0)
  if prefer_orig_order:
    f1 = -k * (
        1. -
        l0 * jnp.array([jnp.sign(dx[0]), jnp.ones_like(dx[1])]) / l) * dx
  else:
    f1 = -k * (1. - l0 / l) * dx
  f1 = jnp.nan_to_num(f1, copy=False, posinf=0., neginf=0.)
  f1p = jnp.pad(f1, ((0, 0), (0, 0), (0, 0), (1, 0)))
  f1n = jnp.pad(f1, ((0, 0), (0, 0), (0, 0), (0, 1)))

  # | springs
  dx = x[..., 1:, :] - x[..., :-1, :] + _xy_vec(0, l0)
  l = jnp.linalg.norm(dx, axis=0)
  if prefer_orig_order:
    f2 = -k * (1. - l0 * jnp.array([jnp.ones_like(dx[0]),
                                    jnp.sign(dx[1])]) / l) * dx
  else:
    f2 = -k * (1. - l0 / l) * dx
  f2 = jnp.nan_to_num(f2, copy=False, posinf=0., neginf=0.)
  f2p = jnp.pad(f2, ((0, 0), (0, 0), (1, 0), (0, 0)))
  f2n = jnp.pad(f2, ((0, 0), (0, 0), (0, 1), (0, 0)))

  # We want to keep elasticity E constant, and k ~ E/l.
  k2 = k / jnp.sqrt(2.0)

  # \ springs
  dx = x[:, :, 1:, 1:] - x[:, :, :-1, :-1] + _xy_vec(l0, l0)
  l = jnp.linalg.norm(dx, axis=0)
  if prefer_orig_order:
    f3 = -k2 * (1. - l0_diag *
                jnp.array([jnp.sign(dx[0]), jnp.sign(dx[1])]) / l) * dx
  else:
    f3 = -k2 * (1. - l0_diag / l) * dx
  f3 = jnp.nan_to_num(f3, copy=False, posinf=0., neginf=0.)
  f3p = jnp.pad(f3, ((0, 0), (0, 0), (1, 0), (1, 0)))
  f3n = jnp.pad(f3, ((0, 0), (0, 0), (0, 1), (0, 1)))

  # / springs
  dx = x[:, :, 1:, :-1] - x[:, :, :-1, 1:] + _xy_vec(-l0, l0)
  l = jnp.linalg.norm(dx, axis=0)
  if prefer_orig_order:
    f4 = -k2 * (1. - l0_diag *
                jnp.array([-jnp.sign(dx[0]), jnp.sign(dx[1])]) / l) * dx
  else:
    f4 = -k2 * (1. - l0_diag / l) * dx
  f4 = jnp.nan_to_num(f4, copy=False, posinf=0., neginf=0.)
  f4p = jnp.pad(f4, ((0, 0), (0, 0), (1, 0), (0, 1)))
  f4n = jnp.pad(f4, ((0, 0), (0, 0), (0, 1), (1, 0)))

  return f1p + f2p + f3p + f4p - f1n - f2n - f3n - f4n


MESH_LINK_DIRECTIONS = (  # xyz
    # 6 nearest neighbors
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    # 12 next-nearest neighbors
    (1, 1, 0),
    (-1, 1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (0, 1, 1),
    (0, -1, 1),
    # 8 next-next-nearest neighors
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (-1, 1, 1))


def elastic_mesh_3d(x: jnp.ndarray,
                    k: float,
                    stride: Union[float, Sequence[float]],
                    prefer_orig_order=False,
                    links=MESH_LINK_DIRECTIONS) -> jnp.ndarray:
  """Computes internal forces on the nodes of a 3d spring mesh.

  Args:
    x: [3, z, y, x] array of mesh node positions, in relative format
    k: spring constant for springs along the x direction; will be scaled
      according to `stride` for all other springs to maintain constant
      elasticity
    stride: XYZ stride of the spring mesh grid
    prefer_orig_order: only False is supported
    links: sequence of XYZ tuples indcating node links to consider, relative to
      the node at (0, 0, 0); valid component values are {-1, 0, 1}

  Returns:
    [3, z, y, x] array of forces
  """
  assert x.ndim == 4
  assert x.shape[0] == 3
  if prefer_orig_order:
    raise NotImplementedError('prefer_orig_order not supported for 3d mesh.')

  if not isinstance(stride, collections.abc.Sequence):
    stride = (stride,) * 3

  stride = np.array(stride)
  f_tot = None
  for direction in links:
    l0 = np.array(stride * direction).reshape([3, 1, 1, 1])
    sel1 = [np.s_[:]]
    sel2 = [np.s_[:]]
    pad_neg = [(0, 0)]
    pad_pos = [(0, 0)]
    for dim in direction[::-1]:  # zyx
      if dim == -1:
        sel1.append(np.s_[:-1])
        sel2.append(np.s_[1:])
        pad_pos.append((0, 1))
        pad_neg.append((1, 0))
      elif dim == 1:
        sel1.append(np.s_[1:])
        sel2.append(np.s_[:-1])
        pad_pos.append((1, 0))
        pad_neg.append((0, 1))
      elif dim == 0:
        sel1.append(np.s_[:])
        sel2.append(np.s_[:])
        pad_pos.append((0, 0))
        pad_neg.append((0, 0))
      else:
        raise ValueError('Only |v| <= 1 values supported within links.')

    dx = x[tuple(sel1)] - x[tuple(sel2)] + l0
    l0 = np.linalg.norm(l0)
    l = jnp.linalg.norm(dx, axis=0)
    f = -k * l0 / stride[0] * (1. - l0 / l) * dx
    f = jnp.nan_to_num(f, copy=False, posinf=0., neginf=0.)
    fp = jnp.pad(f, pad_pos)
    if f_tot is None:
      f_tot = fp
    else:
      f_tot += fp
    fn = jnp.pad(f, pad_neg)
    f_tot -= fn

  return f_tot


@dataclasses_json.dataclass_json
@functools.partial(dataclasses.dataclass, frozen=True)
class IntegrationConfig:
  """Parameters for numerical integration of the mesh state."""

  dt: float  # time step size
  gamma: float  # damping constant
  k0: float  # spring constant for inter-section springs
  k: float  # spring constant for intra-section springs
  stride: float  # distance between nearest neighbors of the point grid
  num_iters: int  # number of time steps to execute at once
  max_iters: int  # upper bound for simulation time

  # The simulation terminates when the velocity of all nodes is below this
  # value. If FIRE is used, the force cap also has to have reached 'final_cap'
  # as specified below in order for the simulation to stop.
  stop_v_max: float

  # Whether to use the Fast Inertial Relaxation Engine (FIRE).
  fire: bool = True

  # FIRE parameters.
  f_alpha: float = 0.99
  f_inc: float = 1.1
  f_dec: float = 0.5
  alpha: float = 0.1
  n_min: int = 5  # Min. number of steps after which to increase step size.
  dt_max: float = 10.  # Max time step size, in units of 'dt'.

  # Initial and final values of the inter-section force component magnitude cap.
  # start_cap != final_cap is only supported when using FIRE.
  start_cap: float = 1e6
  final_cap: float = 1e6

  # Upscaling factor for the force cap (should be >1).
  cap_scale: float = 1.1

  # Number of steps between force cap upscalings. The cap is only updated
  # when power has remained positive for this number of steps. The cap will
  # also be increased once the velocity of all nodes is below 'stop_v_max',
  # regardless of the power history.
  cap_upscale_every: int = 100

  # Whether to modify the in-plane spring force formulation to energetically
  # favor the original relative spatial node ordering. This helps prevent
  # mesh folds, but adds some computational overhead.
  prefer_orig_order: bool = False

  # If true, removes global drift after every step -- subtracts global mean
  # speed from every node, and translates all nodes so that their mean global
  # position is at 0.
  remove_drift: bool = False


@functools.partial(jax.jit, static_argnames=['config', 'mesh_force', 'prev_fn'])
def velocity_verlet(x: jnp.ndarray,
                    v: jnp.ndarray,
                    prev: Optional[jnp.ndarray],
                    config: IntegrationConfig,
                    force_cap: float,
                    fire_dt=None,
                    fire_alpha=None,
                    mesh_force=inplane_force,
                    prev_fn=None):
  """Executes a sequence of (damped) velocity Verlet steps.

  Optionally uses the FIRE integrator. Disabling or reducing
  damping ('gamma') is recommended when using FIRE.

  As of Apr 2021, this runs at 1.3 GLUPS - 2.3 GLUPS. The latter value
  is attained for specific sizes of the arrays (e.g. 1024^2, 2048^2).

  LUPS = Lattice (site) Updates Per Second

  References:
   * derivation of velocity Verlet for damped systems:
     http://physics.bu.edu/py502/lectures3/cmotion.pdf
   * FIRE:
     https://doi.org/10.1103/PhysRevLett.97.170201
     https://doi.org/10.1016/j.commatsci.2018.09.049

  N below indicates the number of degrees of freedom of every node. Valid values
  are 2 (in-plane) or 3 (full 3d). The order of vector components indexed by the
  first dimension is XY[Z].

  Args:
    x: [N, z, y, x] positions
    v: [N, z, y, x] velocities
    prev: [N, z, y, x] positions against which to compute 0-length spring forces
    config: integration parameters
    force_cap: max. magnitude of an inter-section force component
    fire_dt: initial step size when using the FIRE solver; config.dt is used
      when None
    fire_alpha: initial value of alpha when using the FIRE solver; config.alpha
      is used when None
    mesh_force: callable with the signature of `inplane_force` returning a field
      representing internal mesh forces
    prev_fn: callable taking the 'x' mesh array and returning the 'prev' array

  Returns:
    updated mesh state; this is a tuple of:
      position, velocity, acceleration, (dt, alpha, steps since power < 0,
      inter-section force cap)

    The latter 4 entries are only present when using FIRE.
  """

  # The code assumes uniform masses set to unity, so force=acceleration.
  def _force(x, prev, cap):
    a = mesh_force(x, config.k, config.stride, config.prefer_orig_order)
    if prev_fn is not None:
      prev = prev_fn(x)

    if prev is not None:
      a += jnp.clip(-config.k0 * jnp.nan_to_num(x - prev), -cap, cap)
    return a

  def vv_step(t, state, dt, force_cap):
    del t
    x, v, a = state
    x += dt * v + 0.5 * dt**2 * a
    a_prev = a
    a = _force(x, prev, force_cap)

    fact0 = 1.0 / (1.0 + 0.5 * dt * config.gamma)
    fact1 = (1.0 - 0.5 * dt * config.gamma)
    v = fact0 * (v * fact1 + 0.5 * dt * (a_prev + a))
    return x, v, a

  def fire_step(t, state):
    x, v, a, dt, alpha, n_pos, cap = state
    x, v, a = vv_step(t, (x, v, a), dt, cap)

    a_norm = jnp.linalg.norm(a, axis=0, keepdims=True) + 1e-6
    v_norm = jnp.linalg.norm(v, axis=0, keepdims=True)

    power = jnp.vdot(a, v)
    v += alpha * (a / a_norm * v_norm - v)

    # Number of steps since power was negative.
    n_pos = jnp.where(power >= 0, n_pos + 1, 0)

    # FIRE adaptive time stepping scheme:
    # - when power < 0, reduce dt, reset alpha and set v = 0.
    # - when power > 0 for n_min steps, increase dt and alpha.
    dt = jnp.where(
        power >= 0,
        jnp.where(
            n_pos > config.n_min,  #
            jnp.minimum(dt * config.f_inc, config.dt_max * config.dt),
            dt),
        dt * config.f_dec)
    alpha = jnp.where(
        power >= 0,
        jnp.where(n_pos > config.n_min, alpha * config.f_alpha, alpha),
        config.alpha)

    cap = jnp.minimum(
        jnp.where(
            power >= 0,
            jnp.where((n_pos > 0) & ((n_pos % config.cap_upscale_every) == 0),
                      config.cap_scale * cap, cap),  #
            cap),
        config.final_cap)

    v *= (power >= 0)

    if config.remove_drift:
      # Remove any global drift and recenter the nodes.
      x -= jnp.mean(x, axis=(1, 2, 3), keepdims=True)
      v -= jnp.mean(v, axis=(1, 2, 3), keepdims=True)

    return x, v, a, dt, alpha, n_pos, cap

  a = _force(x, prev, force_cap)

  if config.fire:
    if fire_alpha is None:
      fire_alpha = config.alpha
    if fire_dt is None:
      fire_dt = config.dt

    return jax.lax.fori_loop(0, config.num_iters, fire_step,
                             (x, v, a, fire_dt, fire_alpha, 0, force_cap))
  else:
    return jax.lax.fori_loop(
        0, config.num_iters,
        functools.partial(vv_step, dt=config.dt, force_cap=force_cap),
        (x, v, a))


def relax_mesh(
    x: jnp.ndarray,
    prev: Optional[jnp.ndarray],
    config: IntegrationConfig,
    mesh_force=inplane_force,
    prev_fn=None) -> Tuple[jnp.ndarray, List[float], int]:
  """Simulates mesh relaxation.

  Args:
    x: [2, z, y, x] array of mesh node positions
    prev: optional [2, z, y, x] array against which to compute the force due to
      0-length springs
    config: simulation parameters
    mesh_force: callable with the signature of `inplane_force` returning a field
      representing internal mesh forces
    prev_fn: callable taking the 'x' mesh array and returning the 'prev' array

  Returns:
    tuple of:
      [2, z, x, y] array of updated mesh positions
      list of kinetic energy history
      number of simulation steps executed
  """

  t = 0
  v = jnp.zeros_like(x)
  dt = config.dt
  alpha = config.alpha
  e_kin = []
  cap = config.start_cap

  if config.start_cap != config.final_cap:
    if not config.fire:
      raise NotImplementedError(
          'Adaptive force capping is only supported with FIRE.')
    if config.cap_scale <= 1:
      raise ValueError('The scaling factor for the force cap has to be larger '
                       'than 1 when the initial and final cap are different.')

  if prev is not None and prev_fn is not None:
    raise ValueError('Only one of: "prev" and "prev_fn" can be specified.')

  while t < config.max_iters:
    state = velocity_verlet(
        x,
        v,
        prev,
        config,
        fire_dt=dt,
        fire_alpha=alpha,
        force_cap=cap,
        mesh_force=mesh_force,
        prev_fn=prev_fn)
    t += config.num_iters
    x, v = state[:2]
    v_mag = jnp.linalg.norm(v, axis=0)
    e_kin.append(float(jnp.sum(v_mag**2)))
    v_max = jnp.max(v_mag)

    if config.fire:
      dt, alpha, n_pos, cap = state[-4:]
      logging.info(
          't=%r: dt=%f, alpha=%f, n_pos=%d, cap=%f, v_max=%f, e_kin=%f', t, dt,
          alpha, n_pos, cap, v_max, e_kin[-1])

    if v_max < config.stop_v_max:
      if cap >= config.final_cap:
        break

      # Increase cap to ensure progress towards the termination condition.
      cap = min(cap * config.cap_scale, config.final_cap)

  return x, e_kin, t
