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

"""Utilities for manipulating flow arrays.

A flow field has the same physical representation as a relative coordinate map
(see map_utils.py). Flow vectors can have additional statistics associated
with them. When present, these are stored in channels 2+ of the array.

Flow entries can be invalid (i.e., unknown for a given point), in which
case they are marked by nan stored in both X and Y channels.
"""

from typing import Sequence

import numpy as np
from scipy import ndimage


def apply_mask(flow, mask):
  for i in range(flow.shape[0]):
    flow[i, ...][mask] = np.nan


def clean_flow(flow: np.ndarray, min_peak_ratio: float,
               min_peak_sharpness: float, max_magnitude: float,
               max_deviation: float, dim: int = 2) -> np.ndarray:
  """Removes flow vectors that do not fulfill quality requirements.

  Args:
    flow: [c, z, y, x] flow field
    min_peak_ratio: min. value of peak intensity ratio (chan 3); only
      applies to 4-channel input flows
    min_peak_sharpness: min. value of the peak sharpness (chan 2); only
      applies to 4-channel input flows
    max_magnitude: maximum magnitude of a flow component; when <= 0,
      the constraint is not applied
    max_deviation: maximum absolute deviation from the 3x3-window median of a
      flow component; when <= 0, the constraint is not applied
    dim: number of spatial dimensions of the flow field

  Returns:
    filtered flow field in a [2 or 3, z, y, x] array; 3 output channels
    are used only when the input array also has c=3
  """
  assert dim in (2, 3)
  assert dim <= flow.shape[0] <= dim + 2
  if flow.shape[0] == dim + 2:
    ret = flow[:dim, ...].copy()
    bad = np.abs(flow[dim, ...]) < min_peak_sharpness
    pr = np.abs(flow[dim + 1, ...])
    bad |= (pr > 0.0) & (pr < min_peak_ratio)
  else:
    ret = flow.copy()
    bad = np.zeros(flow[0, ...].shape, dtype=bool)

  if max_magnitude > 0:
    bad |= np.max(np.abs(flow[:dim, ...]), axis=0) > max_magnitude

  if max_deviation > 0:
    size = (1, 1, 3, 3) if dim == 2 else (1, 3, 3, 3)
    med = ndimage.median_filter(np.nan_to_num(flow[:dim, ...]), size=size)
    bad |= (np.max(np.abs(med - flow[:dim, ...]), axis=0) > max_deviation)

  apply_mask(ret, bad)
  return ret


def reconcile_flows(flows: Sequence[np.ndarray], max_gradient: float,
                    max_deviation: float, min_patch_size: int,
                    min_delta_z: int = 0) -> np.ndarray:
  """Reconciles multiple flows.

  Args:
    flows: sequence of [c, z, y, x] flow arrays, sorted in order of decreasing
      preference; 'c' can be 2 or 3
    max_gradient: maximum absolute value of the gradient of a flow component;
      when <= 0, the constraint is not applied
    max_deviation: maximum absolute deviation from the 3x3-window median of a
      flow component; when <= 0, the constraint is not applied
    min_patch_size: minimum size of a connected component of the flow field
      in pixels; when <= 0, the constraint is not applied
    min_delta_z: for 3-channel flows, the minimum absolute value of the z
      offset at which flow data is considered valid

  Returns:
    reconciled flow field in a [c, z, y, x] array
  """
  ret = flows[0].copy()
  assert ret.shape[0] in (2, 3)
  for _, f in enumerate(flows[1:]):
    # Try to fill any invalid values.
    m = np.repeat(np.isnan(ret[0:1, ...]), ret.shape[0], 0)
    if ret.shape[0] == 3:
      m &= np.repeat(f[2:3, ...] >= min_delta_z, 3, 0)
    ret[m] = f[m]

  if max_gradient > 0:
    # Invalidate regions where the gradient is too large.
    m = np.abs(np.diff(ret[0, ...], axis=-1, prepend=0)) > max_gradient
    m |= np.abs(np.diff(ret[0, ...], axis=-1, append=0)) > max_gradient
    m |= np.abs(np.diff(ret[1, ...], axis=-2, prepend=0)) > max_gradient
    m |= np.abs(np.diff(ret[1, ...], axis=-2, append=0)) > max_gradient
    apply_mask(ret, m)

  # Filter out points that deviate too much from the median. This gets rid
  # of small, few-point anomalies.
  if max_deviation > 0:
    med = ndimage.median_filter(np.nan_to_num(ret), size=(1, 1, 3, 3))
    bad = (np.max(np.abs(med - ret)[:2, ...], axis=0) > max_deviation)
    apply_mask(ret, bad)

  if min_patch_size > 0:
    bad = np.zeros(ret[0, ...].shape, dtype=bool)
    valid = ~np.any(np.isnan(ret), axis=0)
    for z in range(valid.shape[0]):
      labeled, _ = ndimage.label(valid[z, ...])
      ids, sizes = np.unique(labeled, return_counts=True)
      small = ids[sizes < min_patch_size]
      bad[z, ...][np.in1d(labeled.ravel(), small).reshape(labeled.shape)] = True
    apply_mask(ret, bad)

  return ret
