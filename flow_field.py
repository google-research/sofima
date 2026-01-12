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
"""Utilities for calculating flow-fields over images.

The flow fields have single pixel precision (i.e. the calculated
flow vectors have integer components).
"""

import collections
import functools
from typing import Callable, Iterator, Sequence, TypeVar
from absl import logging
from connectomics.common import geom_utils
from connectomics.common import utils
import jax
import jax.numpy as jnp
import numpy as np
import scipy.fftpack

Array = np.ndarray | jnp.ndarray
T = TypeVar('T')


def masked_xcorr(
    prev: Array,
    curr: Array,
    prev_mask: Array | None = None,
    curr_mask: Array | None = None,
    use_jax: bool = False,
    dim: int = 2,
) -> Array:
  """Computes cross-correlation between two masked images.

  Correlation is computed over the last 'dim' dimensions. The remaining
  dimensions, if any, are treated as batch.

  Args:
    prev: 1st image to cross-correlate
    curr: 2nd image to cross-correlate
    prev_mask: optional mask indicating invalid pixels in the 1st image
    curr_mask: optional mask indicating invalid pixels in the 2nd image
    use_jax: whether to use JAX for the computations
    dim: number of spatial dimensions in the images and masks (last N axes)

  Returns:
    cross-correlation data between the two inputs (same spatial dimensionality
    as the inputs)

  The implementation follows the method described in:
    D. Padfield, Masked Object Registration in the Fourier Domain,
    http://dx.doi.org/10.1109/TIP.2011.2181402
  """
  xnp = jnp if use_jax else np
  shape = np.array(prev.shape[-dim:]) + np.array(curr.shape[-dim:]) - 1
  fast_shape = [scipy.fftpack.next_fast_len(int(x)) for x in shape]
  out_slice = tuple(
      [slice(None)] * (len(prev.shape) - dim)
      + [slice(0, int(x)) for x in shape]
  )

  if prev_mask is not None:
    prev = xnp.where(prev_mask, 0.0, prev)
  if curr_mask is not None:
    curr = xnp.where(curr_mask, 0.0, curr)

  slc = np.index_exp[...] + np.index_exp[::-1] * dim
  curr = curr[slc]

  fft = functools.partial(xnp.fft.rfftn, s=fast_shape)
  ifft = functools.partial(xnp.fft.irfftn, s=fast_shape)
  p_f = fft(prev)
  c_f = fft(curr)
  xcorr = ifft(p_f * c_f)

  # Save unnecessary computation if no mask is provided.
  if prev_mask is None and curr_mask is None:
    return xcorr[out_slice]

  if prev_mask is None:
    prev_mask = xnp.ones(prev.shape, dtype=bool)
  else:
    prev_mask = xnp.logical_not(prev_mask)

  if curr_mask is None:
    curr_mask = xnp.ones(curr.shape, dtype=bool)
  else:
    curr_mask = xnp.logical_not(curr_mask)

  curr_mask = curr_mask[slc]

  pm_f = fft(prev_mask)
  cm_f = fft(curr_mask)

  def fmax(x, v=0.0):
    if use_jax:
      return jnp.fmax(x, v)
    else:
      np.fmax(x, v, out=x)
      return x

  eps = xnp.finfo(xnp.float32).eps
  overlap_masked_px = xnp.round(ifft(cm_f * pm_f))
  overlap_masked_px = fmax(overlap_masked_px, eps)
  overlap_inv = 1.0 / overlap_masked_px

  mc_p_f = ifft(cm_f * p_f)
  mc_c_f = ifft(pm_f * c_f)

  xcorr -= mc_p_f * mc_c_f * overlap_inv

  p_sq_f = fft(xnp.square(prev))
  p_denom = ifft(cm_f * p_sq_f) - xnp.square(mc_p_f) * overlap_inv
  p_denom = fmax(p_denom)

  c_sq_f = fft(xnp.square(curr))
  c_denom = ifft(pm_f * c_sq_f) - xnp.square(mc_c_f) * overlap_inv
  c_denom = fmax(c_denom)

  denom = xnp.sqrt(p_denom * c_denom)

  xcorr = xcorr[out_slice]
  denom = denom[out_slice]
  overlap_masked_px = overlap_masked_px[out_slice]

  tol = 1e3 * eps * xnp.max(xnp.abs(denom), keepdims=True)
  nonzero_indices = denom > tol

  if use_jax:
    out = jnp.where(nonzero_indices, xcorr / denom, 0.0)
  else:
    out = np.zeros_like(denom)
    out[nonzero_indices] = xcorr[nonzero_indices] / denom[nonzero_indices]

  if use_jax:
    out = jnp.clip(out, min=-1, max=1)
  else:
    np.clip(out, min=-1, max=1, out=out)

  px_threshold = 0.3 * xnp.max(overlap_masked_px, keepdims=True)
  if use_jax:
    out = jnp.where(overlap_masked_px < px_threshold, 0.0, out)
  else:
    out[overlap_masked_px < px_threshold] = 0.0
  return out


@jax.jit
def _integral_image(mask: jax.Array | None) -> np.ndarray | jax.Array | None:
  """Computes a summed volume table of 'val'."""

  if mask is None:
    return mask

  if mask.size >= 2**32:
    return geom_utils.integral_image(np.array(mask).astype(np.int64))

  pads = []
  ii = jnp.array(mask.astype(np.uint32))
  for axis in range(len(mask.shape)):
    ii = ii.cumsum(axis=axis)
    pads.append([1, 0])

  return jnp.pad(ii, pads, mode='constant')


def _peak_stats(peak1_val, peak2_val, peak1_idx, img, offset, peak_radius=5):
  """Computes peak quality statistics."""
  dim = len(offset)
  inds = jnp.unravel_index(peak1_idx, img.shape[-dim:])
  center_inds = [
      x.astype(jnp.float32) - offset for x, offset in zip(inds, offset)
  ]

  if not isinstance(peak_radius, collections.abc.Sequence):
    peak_radius = (peak_radius,) * dim

  peak_radius = np.array(peak_radius)
  size = 2 * peak_radius + 1
  start = jnp.asarray(inds) - size // 2
  sharpness = img[inds] / jnp.min(jax.lax.dynamic_slice(img, start, size))

  return jnp.where(
      jnp.isinf(peak1_val),  #
      jnp.array([jnp.nan] * (dim + 2)),
      jnp.where(
          jnp.isinf(peak2_val),  #
          jnp.array(center_inds[::-1] + [sharpness, 0.0]),
          jnp.array(center_inds[::-1] + [sharpness, peak1_val / peak2_val]),
      ),
  )


def _batched_peaks(
    img: jnp.ndarray,
    center_offset: jnp.ndarray,
    min_distance: int | Sequence[int],
    threshold_rel: float,
    peak_radius: int | Sequence[int] = 5,
) -> jnp.ndarray:
  """Computes peak statistics from a batch of correlation images.

  Args:
    img: [b, [z,] y, x] correlation images
    center_offset: ([z,] y, x) peak location within a correlation image computed
      for two aligned patches (with no relative shift)
    min_distance: min. distance in pixels between peaks
    threshold_rel: min. fraction of the max of the input image that the
      intensity at a prospective peak needs to exceed
    peak_radius: radius to use for peak sharpness calculation

  Returns:
    [b, 4 or 5] array of peak locations and statistics; the values in the
    last dim. are as follows:
      x, y [, z] peak offset from center,
      peak sharpness indicator (higher absolute value is more sharp)
      peak height ratio between two largest peaks, if more than 1
        peak is identified; nan otherwise
  """
  dim = img.ndim - 1
  if isinstance(min_distance, collections.abc.Sequence):
    assert len(min_distance) == dim
  else:
    size = [2 * min_distance + 1] * dim

  # Apply the maximum filter as a sequence of 1d filters.
  img_max = img
  strides = (1,) * dim
  for i, s in enumerate(size):
    patch = [1] * dim
    patch[i] = s
    img_max = jnp.max(
        jax.lax.conv_general_dilated_patches(
            img_max[:, np.newaxis, ...], patch, strides, 'same'
        ),
        axis=1,
    )

  # Create a boolean mask with the peaks.
  thresholds = threshold_rel * img.max(
      axis=tuple(range(-dim, 0)), keepdims=True
  )
  peak_mask = (img == img_max) & (img > thresholds)
  masked_flat_img = jnp.where(peak_mask, img, -jnp.inf)
  masked_flat_img = masked_flat_img.reshape(peak_mask.shape[0], -1)

  # Find the location and value of the top two peaks.
  max_peaks_1st = jnp.argmax(masked_flat_img, axis=-1)
  peak_vals_1st = jnp.take_along_axis(
      masked_flat_img, max_peaks_1st[:, np.newaxis], axis=-1
  )[:, 0]
  max_peaks_2nd = jnp.argmax(
      masked_flat_img.at[:, max_peaks_1st].set(-jnp.inf), axis=-1
  )
  peak_vals_2nd = jnp.take_along_axis(
      masked_flat_img, max_peaks_2nd[:, np.newaxis], axis=-1
  )[:, 0]

  peak_stats_fn = functools.partial(
      _peak_stats, offset=center_offset, peak_radius=peak_radius
  )
  return jax.vmap(peak_stats_fn)(
      peak_vals_1st, peak_vals_2nd, max_peaks_1st, img
  )


def _batched_xcorr(
    pre_image: jnp.ndarray,
    post_image: jnp.ndarray,
    pre_mask: jnp.ndarray | None,
    post_mask: jnp.ndarray | None,
    patch_size: Sequence[int],
    starts: jnp.ndarray,
    mean: float | None,
    post_patch_size: Sequence[int] | None = None,
    post_starts: jax.Array | None = None,
) -> tuple[np.ndarray, jnp.ndarray]:
  """Extracts a batch of patch pairs and cross-correlates them.

  Patches are extracted from matching locations in the pre/post images.

  Args:
    pre_image: [[z,] y, x] 1st image array from which to extract patches
    post_image: [[z,] y, x] 2nd image array from which to extract patches
    pre_mask: [[z,] y, x] mask for the 1st image
    post_mask: [[z,] y, x] mask for the 2nd image
    patch_size: ([z,] y, x) size of the patches to extract
    starts: [b, 2 or 3] array of top-left ([z]yx) coordinates of the patches to
      extract
    mean: value to subtract from patches; if None, per-patch arithmetic mean
      will be used
    post_patch_size: ([z,] y, x) size of patches to extract from 'post_image';
      if not specified, 'patch_size' is used
    post_starts: [b, 2 or 3] array of top-left ([z]yx) coordinates of the
      patches to extract from the 'post' image

  Returns:
    tuple of:
      location of the peak in the cross-correlation image corresponding to
        null shift
      [b, [z',] y', x'] array of cross-correlation images
  """
  if post_patch_size is None:
    post_patch_size = patch_size

  if post_starts is None:
    post_starts = starts

  pre_batch = jax.vmap(
      lambda x: jax.lax.dynamic_slice(pre_image, x, patch_size)
  )(starts)
  post_batch = jax.vmap(
      lambda x: jax.lax.dynamic_slice(post_image, x, post_patch_size)
  )(post_starts)
  if pre_mask is None:
    pre_mask_batch = None
  else:
    pre_mask_batch = jax.vmap(
        lambda x: jax.lax.dynamic_slice(pre_mask, x, patch_size)
    )(starts)

  if post_mask is None:
    post_mask_batch = None
  else:
    post_mask_batch = jax.vmap(
        lambda x: jax.lax.dynamic_slice(post_mask, x, post_patch_size)
    )(post_starts)

  def _masked_mean(source, mask):
    axes = tuple(range(-1, -(len(patch_size) + 1), -1))
    if mask is None:
      return jnp.mean(source, axis=axes, keepdims=True)
    else:
      return jnp.nanmean(
          jnp.where(mask, jnp.nan, source), axis=axes, keepdims=True
      )

  if mean is None:
    pre_mean = _masked_mean(pre_batch, pre_mask_batch)
    post_mean = _masked_mean(post_batch, post_mask_batch)
  else:
    pre_mean = post_mean = mean

  # Where the peak should be in the output xcorr array if there is no
  # section-to-section shift.
  center_offset = (
      np.array(pre_batch.shape[-len(patch_size) :])
      + post_batch.shape[-len(patch_size) :]
  ) // 2 - 1
  return (
      center_offset,  # pytype: disable=bad-return-type  # jax-ndarray
      masked_xcorr(
          pre_batch - pre_mean,
          post_batch - post_mean,
          pre_mask_batch,
          post_mask_batch,
          use_jax=True,
          dim=len(patch_size),
      ),
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        'patch_size',
        'mean',
        'min_distance',
        'threshold_rel',
        'peak_radius',
        'post_patch_size',
    ],
)
def batched_xcorr_peaks(
    pre_image: jnp.ndarray,
    post_image: jnp.ndarray,
    pre_mask: jnp.ndarray | None,
    post_mask: jnp.ndarray | None,
    patch_size: Sequence[int],
    starts: jnp.ndarray,
    mean: float | None,
    min_distance: int | Sequence[int] = 2,
    threshold_rel: float = 0.5,
    peak_radius: int | Sequence[int] = 5,
    post_patch_size: Sequence[int] | None = None,
    post_starts: jax.Array | None = None,
) -> jnp.ndarray:
  """Computes cross-correlations and identifies their peaks.

  Args:
    pre_image: [[z,] y, x] 1st image array from which to extract patches
    post_image: [[z, ]y, x] 2nd image array from which to extract patches
    pre_mask: [[z,] y, x] mask for the 1st image
    post_mask: [[z,] y, x] mask for the 2nd image
    patch_size: ([z,] y, x) size of the patches to extract
    starts: [b, 2 or 3] array of top-left ([z]yx) coordinates of the patches to
      extract
    mean: value to subtract from patches; if None, per-patch arithmetic mean
      will be used
    min_distance: min. distance in pixels between peaks
    threshold_rel: min. fraction of the max value in the input image that the
      intensity at a prospective peak needs to exceed
    peak_radius: radius to use for peak shaprness calculation
    post_patch_size: ([z,] y, x) size of patches to extract from 'post_image';
      if not specified, 'patch_size' is used
    post_starts: [b, 2 or 3] array of top-left ([z]yx) coordinates of the
      patches to extract from the 'post' image

  Returns:
    [b, 4 or 5] array of peak information; see _batched_peaks for details
  """
  center_offset, xcorr = _batched_xcorr(
      pre_image,
      post_image,
      pre_mask,
      post_mask,
      patch_size,
      starts,
      mean,
      post_patch_size,
      post_starts,
  )
  peaks = _batched_peaks(
      xcorr,
      center_offset,
      min_distance,
      threshold_rel,  # pytype: disable=wrong-arg-types  # jax-ndarray
      peak_radius,
  )
  return peaks


def _silent_fn(x: list[T]) -> Iterator[T]:
  for item in x:
    yield item


class JAXMaskedXCorrWithStatsCalculator:
  """Estimates optical flow using masked cross-correlation.

  Computes the flow field entries in batches for improved performance.
  """

  non_spatial_flow_channels = 2  # peak sharpness, peak ratio

  def __init__(
      self,
      mean: float | None = None,
      peak_min_distance: float = 2,
      peak_radius: float = 5,
  ):
    """Constructor.

    Args:
      mean: optional mean value to subtract from the patches
      peak_min_distance: min. distance in pixels between peaks
      peak_radius: radius to use for peak shaprness calculation
    """
    self._mean = mean
    self._min_distance = peak_min_distance
    self._peak_radius = peak_radius

  def flow_field(
      self,
      pre_image: np.ndarray,
      post_image: np.ndarray,
      patch_size: int | Sequence[int],
      step: int | Sequence[int],
      pre_mask=None,
      post_mask=None,
      mask_only_for_patch_selection=False,
      selection_mask=None,
      max_masked=0.75,
      batch_size=4096,
      post_patch_size: int | Sequence[int] | None = None,
      pre_targeting_field: np.ndarray | None = None,
      pre_targeting_step: int | Sequence[int] | None = None,
      post_targeting_field: np.ndarray | None = None,
      post_targeting_step: int | Sequence[int] | None = None,
      progress_fn: Callable[[list[T]], Iterator[T]] = _silent_fn,
  ):
    """Computes the flow field from post to pre.

    The flow is computed using masked cross-correlation.

    Args:
      pre_image: 1st n-dim image input to cross-correlation (yx)
      post_image: 2nd n-dim image input for cross-correlation (yx)
      patch_size: size of patches to correlate (yx)
      step: step at which to sample patches to correlate (yx)
      pre_mask: optional mask for the 1st image (yx)
      post_mask: optional mask for the 2nd image (yx)
      mask_only_for_patch_selection: whether to only use mask to decide for
        which patch pairs to compute flow
      selection_mask: optional mask of the same shape as the output flow field,
        in which positive entries indicate flow entries that need to be
        computed; locations corresponding to negative values in the mask will be
        filled with nans.
      max_masked: max. fraction of image pixels to be masked within a patch for
        the patch to be considered for flow computation
      batch_size: number of patches to process at the same time; larger values
        are generally more efficient
      post_patch_size: patch size to use for 'post_image'; if not specified,
        'patch_size' is used for both pre and post images
      pre_targeting_field: dense xy[z] vector field with precomputed flows used
        to target the location of 'pre' patches; typically from a prior run of
        'flow_field'
      pre_targeting_step: step size at which 'pre_targeting_field' values were
        sampled (same units as 'step', yx order)
      post_targeting_field: like 'pre_targeting_field', but for shifting the
        'post_image' patches
      post_targeting_step: step size at which 'post_targeting_field' values were
        sampled (same units as 'step', yx order)
      progress_fn: function taking a list of batches of 'post' z[yx] start
        positions to process; can be used with tqdm to track progress

    Returns:
      Flow field calculated from 'post' to 'pre' at reduced resolution grid
      locations indicated by '(post_)patch_size' and 'step'. The order of the
      flow vector components is *opposite* of the image dimension order.
    """
    assert pre_image.ndim == post_image.ndim

    if not isinstance(patch_size, collections.abc.Sequence):
      patch_size = (patch_size,) * pre_image.ndim

    if post_patch_size is not None:
      if not isinstance(post_patch_size, collections.abc.Sequence):
        post_patch_size = (post_patch_size,) * post_image.ndim
    else:
      post_patch_size = patch_size

    if not isinstance(step, collections.abc.Sequence):
      step = (step,) * pre_image.ndim

    if pre_targeting_step is not None and not isinstance(
        pre_targeting_step, collections.abc.Sequence
    ):
      pre_targeting_step = (pre_targeting_step,) * pre_image.ndim

    assert len(patch_size) == pre_image.ndim
    assert len(post_patch_size) == post_image.ndim
    assert len(step) == pre_image.ndim

    # Initialize output flow field.
    out_shape = (post_image.shape - (np.array(post_patch_size) - step)) // step
    out_sel = ()
    for s in out_shape:
      out_sel += np.index_exp[:s]

    output = np.full(
        [self.non_spatial_flow_channels + pre_image.ndim] + out_shape.tolist(),
        np.nan,
        dtype=np.float32,
    )

    # Positions corresponding to positive values in the selection mask will
    # be populated with estimated flow vectors.
    if selection_mask is None:
      selection_mask = np.ones(out_shape, dtype=bool)
    else:
      selection_mask = selection_mask[out_sel].copy()

    if pre_mask is not None:
      s = geom_utils.query_integral_image(
          _integral_image(pre_mask), patch_size, step
      )
      m = (s / np.prod(patch_size) >= max_masked)[out_sel]
      selection_mask[m] = False
      del s

    if post_mask is not None:
      s = geom_utils.query_integral_image(
          _integral_image(post_mask), post_patch_size, step
      )
      m = (s / np.prod(post_patch_size) >= max_masked)[out_sel]
      selection_mask[m] = False
      del s

    if mask_only_for_patch_selection:
      pre_mask = post_mask = None
    else:
      if pre_mask is not None:
        pre_mask = jnp.asarray(pre_mask)
      if post_mask is not None:
        post_mask = jnp.asarray(post_mask)

    pre_image = jnp.asarray(pre_image)
    post_image = jnp.asarray(post_image)

    # Offset to add to the starts of the 'prev' patches so that
    # the 'post' patches remain centered at the same location.
    patch_offset = (np.array(patch_size) - post_patch_size) // 2
    patch_offset = patch_offset[None, ...].astype(int)

    oyx = np.array(np.where(selection_mask)).T
    logging.info('Starting flow estimation for %d patches.', oyx.shape[0])

    for pos_zyx in progress_fn(list(utils.batch(oyx, batch_size))):
      pos_zyx = np.array(pos_zyx)
      real_batch_size = pos_zyx.shape[0]

      if real_batch_size < batch_size:
        pad_len = batch_size - real_batch_size
        pos_zyx_proc = np.pad(pos_zyx, ((0, pad_len), (0, 0)), mode='edge')
      else:
        pos_zyx_proc = pos_zyx

      post_starts = pos_zyx_proc * np.array(step).reshape((1, -1))

      pre_starts = post_starts - patch_offset
      pre_starts = np.clip(pre_starts, 0, np.inf).astype(int)

      tg_offsets = None
      if pre_targeting_field is not None and pre_targeting_step is not None:
        pre_center = (np.array(patch_size) // 2).reshape((1, -1))  # [z]yx
        tg_step = np.array(pre_targeting_step).reshape((1, -1))  # [z]yx
        query = np.round((pre_starts + pre_center) / tg_step)
        query = query.astype(int)  # [b, [z]yx]
        q = []
        for i in range(query.shape[-1]):
          q.append(
              np.clip(query[:, i], 0, pre_targeting_field.shape[i + 1] - 1)
          )

        field_indexer = (slice(None),) + tuple(q)
        tg_offsets = np.nan_to_num((pre_targeting_field[field_indexer].T))
        tg_offsets = tg_offsets.astype(int)[:, ::-1]  # [b, xy[z]] -> [b, [z]yx]
        new_starts = pre_starts + tg_offsets

        # Clip offsets that would cause the 'pre' patch to go out of bounds.
        tg_offsets = tg_offsets - np.minimum(new_starts, 0)
        img_shape = np.array(pre_image.shape)[None, ...]
        new_ends = new_starts + np.array(patch_size)[None, ...]
        overshoot = np.maximum(new_ends, img_shape) - img_shape
        tg_offsets = tg_offsets - overshoot

        pre_starts = pre_starts + tg_offsets

      post_offsets = None
      if post_targeting_field is not None and post_targeting_step is not None:
        post_center = (np.array(post_patch_size) // 2).reshape((1, -1))
        tg_step = np.array(post_targeting_step).reshape((1, -1))
        query = np.round((post_starts + post_center) / tg_step)
        query = query.astype(int)  # [b, [z]yx]
        q = []
        for i in range(query.shape[-1]):
          q.append(
              np.clip(query[:, i], 0, post_targeting_field.shape[i + 1] - 1)
          )

        field_indexer = (slice(None),) + tuple(q)
        post_offsets = np.nan_to_num((post_targeting_field[field_indexer].T))
        post_offsets = post_offsets.astype(int)[
            :, ::-1
        ]  # [b, xy[z]] -> [b, [z]yx]
        new_starts = post_starts + post_offsets

        # Clip offsets that would cause the 'post' patch to go out of bounds.
        post_offsets = post_offsets - np.minimum(new_starts, 0)
        img_shape = np.array(post_image.shape)[None, ...]
        new_ends = new_starts + np.array(post_patch_size)[None, ...]
        overshoot = np.maximum(new_ends, img_shape) - img_shape
        post_offsets = post_offsets - overshoot

        post_starts = post_starts + post_offsets

      pre_starts = np.clip(pre_starts, 0, np.inf).astype(int)
      post_starts = np.clip(post_starts, 0, np.inf).astype(int)

      logging.info('.. estimating %d patches.', len(pos_zyx))
      peaks = np.array(
          batched_xcorr_peaks(
              pre_image,
              post_image,
              pre_mask,
              post_mask,
              patch_size,
              jnp.array(pre_starts),
              self._mean,
              post_patch_size=post_patch_size,
              min_distance=self._min_distance,
              peak_radius=self._peak_radius,
              post_starts=jnp.array(post_starts),
          )
      )
      logging.info('.. done.')
      d = pre_image.ndim

      for i, coord in enumerate(pos_zyx):
        v = peaks[i]
        if tg_offsets is not None:
          v[:d] = v[:d] + tg_offsets[i, ::-1]  # xy[z]

        if post_offsets is not None:
          v[:d] = v[:d] - post_offsets[i, ::-1]  # xy[z]

        output[np.index_exp[:] + tuple(coord)] = v

    logging.info('Flow field estimation complete.')
    return output
