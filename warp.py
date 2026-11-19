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
"""Utilities for warping image and point data between coordinate systems."""

from collections.abc import Callable, Sequence
from concurrent import futures
import dataclasses
from typing import Any
from connectomics.common import bounding_box
from connectomics.common import box_generator
from connectomics.segmentation import labels
# pylint:disable=g-import-not-at-top
import cv2 as cv
import numpy as np
from scipy import ndimage
import skimage.exposure
from sofima import map_utils
# pylint:enable=g-import-not-at-top


def _cv_interpolation(inter_scheme: str):
  inter_map = {
      'nearest': cv.INTER_NEAREST,
      'linear': cv.INTER_LINEAR,
      'cubic': cv.INTER_CUBIC,
      'lanczos': cv.INTER_LANCZOS4,
  }
  return inter_map[inter_scheme]


def _relabel_segmentation(data, orig_to_low, old_uids):
  new_uids = frozenset(np.unique(data.astype(np.uint64)))

  # No new IDs are introduced by the warping.
  diff_ids = (new_uids - old_uids) - {0}
  assert not diff_ids, f'Found unexpected new IDs: {diff_ids}'

  orig_ids, low_ids = zip(*orig_to_low)
  return labels.relabel(
      data.astype(np.uint64),
      np.array(low_ids, dtype=np.uint64),
      np.array(orig_ids, dtype=np.uint64),
  )


def _run_parallel(
    fn: Callable[[Any], None], items: Sequence[Any], parallelism: int
) -> None:
  if parallelism > 1 and len(items) > 1:
    with futures.ThreadPoolExecutor(max_workers=parallelism) as exc:
      fs = [exc.submit(fn, item) for item in items]
      for f in futures.as_completed(fs):
        f.result()
  else:
    for item in items:
      fn(item)


def _prepare_cv_image(image: np.ndarray) -> np.ndarray:
  if image.dtype == np.uint32:
    if image.size > 0 and image.max() >= 2**16:
      raise ValueError(
          'Image warping supported up to uint16 only. For segmentation data,'
          ' use uint64.'
      )
    return image.astype(np.uint16)
  return image


@dataclasses.dataclass(frozen=True)
class _CvRemapAxis:
  """1D linear gather indices and weights along an axis with length >= 2."""

  sl: slice
  idx: np.ndarray
  w0: np.ndarray
  w1: np.ndarray


@dataclasses.dataclass(frozen=True)
class _CvRemapGrid:
  """Precomputed 1D separable gather tables for OpenCV remap generation."""

  out_hw: tuple[int, int]
  x_axis: _CvRemapAxis | None
  y_axis: _CvRemapAxis | None


def _prepare_cv_remap_axis(
    m_len: int,
    out_len: int,
    origin: float,
    stride: float,
    weight_shape: tuple[int, int],
) -> _CvRemapAxis | None:
  """Prepares 1D linear gather indices and weights for an axis, or None."""
  if m_len < 2:
    return None
  coords = (np.arange(out_len, dtype=np.float64) - origin) / stride
  k = np.clip(np.floor(coords).astype(np.intp), 0, m_len - 2)
  k_min = int(k.min())
  k_max = int(k.max())
  w1 = (coords - k).reshape(weight_shape)
  return _CvRemapAxis(
      sl=slice(k_min, k_max + 2),
      idx=k - k_min,
      w0=1.0 - w1,
      w1=w1,
  )


def _prepare_cv_remap(
    coord_map: np.ndarray,
    map_box: bounding_box.BoundingBox,
    image_box: bounding_box.BoundingBox,
    out_box: bounding_box.BoundingBox,
    stride: float,
    offset: float = 0.0,
) -> tuple[np.ndarray, _CvRemapGrid]:
  """Converts coord_map to local coordinates and builds 1D remap tables."""
  abs_map = map_utils.to_absolute(coord_map, stride)
  abs_map += (
      map_box.start[:2] * stride - image_box.start[:2] + offset
  ).reshape(2, 1, 1, 1)

  hm, wm = coord_map.shape[2], coord_map.shape[3]
  out_h, out_w = int(out_box.size[1]), int(out_box.size[0])
  grid = _CvRemapGrid(
      out_hw=(out_h, out_w),
      x_axis=_prepare_cv_remap_axis(
          wm,
          out_w,
          float(map_box.start[0] * stride - out_box.start[0] + offset),
          stride,
          (1, -1),
      ),
      y_axis=_prepare_cv_remap_axis(
          hm,
          out_h,
          float(map_box.start[1] * stride - out_box.start[1] + offset),
          stride,
          (-1, 1),
      ),
  )
  return abs_map, grid


def _eval_remap_channel(ch: np.ndarray, grid: _CvRemapGrid) -> np.ndarray:
  y, x = grid.y_axis, grid.x_axis
  sub = ch[
      slice(None) if y is None else y.sl,
      slice(None) if x is None else x.sl,
  ]
  if x is not None:
    sub = sub[:, x.idx] * x.w0 + sub[:, x.idx + 1] * x.w1
  if y is not None:
    sub = sub[y.idx, :] * y.w0 + sub[y.idx + 1, :] * y.w1
  return np.broadcast_to(sub, grid.out_hw).astype(np.float32)


def _compute_cv_remap_tables(
    abs_map_2d: np.ndarray,
    grid: _CvRemapGrid,
    nninterpolation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes fixed-point OpenCV remap tables from a [2, hm, wm] map slice."""
  # dxy: [0 .. out_box.size] -> [coord within image]
  dx = _eval_remap_channel(abs_map_2d[0], grid)
  dy = _eval_remap_channel(abs_map_2d[1], grid)
  return cv.convertMaps(
      dx,
      dy,
      dstmap1type=cv.CV_16SC2,
      nninterpolation=nninterpolation,
  )


def warp_subvolume(
    image: np.ndarray,
    image_box: bounding_box.BoundingBox,
    coord_map: np.ndarray,
    map_box: bounding_box.BoundingBox,
    stride: float,
    out_box: bounding_box.BoundingBox,
    interpolation: str | None = None,
    offset: float = 0.0,
    parallelism: int = 1,
) -> np.ndarray:
  """Warps a subvolume of data according to a coordinate map.

  Args:
    image: [n, z, y, x] data to warp; valid data types are those supported by
      OpenCV's `remap` as well as uint64, which is treated as segmentation data
    image_box: bounding box identifying the part of the volume from which the
      image data was extracted
    coord_map: [2, z, y, x] xy 'inverse' coordinate map in relative format (each
      entry in the map specifies the source coordinate in 'image' from which to
      read data)
    map_box: bounding box identifying the part of the volume from which the
      coordinate map was extracted
    stride: length in pixels of the image corresponding to a single unit (pixel)
      of the coordinate map
    out_box: bounding box for the warped data
    interpolation: interpolation scheme to use; defaults to nearest neighbor for
      uint64 data, and Lanczos for other types
    offset: (deprecated, do not use) non-zero values necessary to reproduce some
      old renders
    parallelism: number of threads to use for warping sections

  Returns:
    warped image covering 'out_box'
  """

  orig_dtype = image.dtype
  warped_shape = [image.shape[0]] + list(out_box.size[::-1])
  if 0 in warped_shape:
    return np.zeros(shape=warped_shape, dtype=orig_dtype)

  all_nan_z = np.all(np.isnan(coord_map), axis=(0, 2, 3))
  active_zs = [z for z in range(image.shape[1]) if not all_nan_z[z]]
  if not active_zs:
    return np.zeros(shape=warped_shape, dtype=orig_dtype)

  # Segmentation warping.
  if orig_dtype == np.uint64:
    interpolation = cv.INTER_NEAREST
    image, orig_to_low = labels.make_contiguous(image)
    assert np.max(image) < 2**31
    assert np.min(image) >= 0
    image = image.astype(np.int32)
    old_uids = frozenset(np.unique(image))
  # Image warping.
  else:
    orig_to_low = None
    old_uids = frozenset()
    if interpolation is None:
      interpolation = cv.INTER_LANCZOS4
    elif isinstance(interpolation, str):
      interpolation = _cv_interpolation(interpolation)
    image = _prepare_cv_image(image)

  # Convert values within the coordinate map to 'image' coordinates and
  # precompute 1D separable gather tables within 'out_box'.
  abs_map, remap_grid = _prepare_cv_remap(
      coord_map, map_box, image_box, out_box, stride, offset
  )

  warped = np.zeros(shape=warped_shape, dtype=image.dtype)

  def _warp_section(z):
    dx, dy = _compute_cv_remap_tables(
        abs_map[:, z, ...],
        remap_grid,
        nninterpolation=(interpolation == cv.INTER_NEAREST),
    )

    for c in range(0, image.shape[0]):
      warped[c, z, ...] = cv.remap(
          image[c, z, ...], dx, dy, interpolation=interpolation
      )

  _run_parallel(_warp_section, active_zs, parallelism)

  # Map IDs back to the original space, which might be beyond the range of
  # int32.
  if orig_to_low is not None:
    warped = _relabel_segmentation(warped, orig_to_low, old_uids)
  else:
    warped = warped.astype(orig_dtype, copy=False)

  return warped


def _separable_linear_upsample(
    eval_coords: np.ndarray, src_coords_1d: Sequence[np.ndarray]
) -> np.ndarray:
  """Separable multilinear upsampling matching ndimage.map_coordinates."""
  out_shape = tuple(q.size for q in src_coords_1d)
  if 0 in out_shape:
    return np.zeros(out_shape, dtype=np.float64)

  axis_params: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = (
      []
  )
  slices: list[slice] = []
  for axis, q_1d in enumerate(src_coords_1d):
    m_len = eval_coords.shape[axis]
    valid = (q_1d >= 0.0) & (q_1d <= float(m_len - 1))
    if not np.any(valid):
      return np.zeros(out_shape, dtype=np.float64)
    if m_len == 1:
      slices.append(slice(0, 1))
      axis_params.append((valid, None, None))
    else:
      q_safe = np.where(valid, q_1d, 0.0)
      idx0 = np.clip(np.floor(q_safe).astype(np.intp), 0, m_len - 2)
      lo = int(idx0[valid].min())
      hi = int(idx0[valid].max()) + 2
      slices.append(slice(lo, hi))
      idx0_local = np.where(valid, idx0 - lo, 0)
      w1_1d = np.where(valid, q_safe - idx0, 0.0)
      axis_params.append((valid, idx0_local, w1_1d))

  cur = eval_coords[tuple(slices)].astype(np.float64, copy=False)
  for axis, (valid, idx0_local, w1_1d) in enumerate(axis_params):
    bcast = [1] * cur.ndim
    bcast[axis] = valid.size
    if idx0_local is None:
      if np.all(valid):
        cur = np.repeat(cur, valid.size, axis=axis)
      else:
        cur = np.where(valid.reshape(bcast), cur, 0.0)
    else:
      assert w1_1d is not None
      w1 = w1_1d.reshape(bcast)
      cur = (
          np.take(cur, idx0_local, axis=axis) * (1.0 - w1)
          + np.take(cur, idx0_local + 1, axis=axis) * w1
      )
      if not np.all(valid):
        cur = np.where(valid.reshape(bcast), cur, 0.0)
  return cur


def ndimage_warp_multi(
    images: Sequence[np.ndarray],
    coord_map: np.ndarray,
    stride: Sequence[float],
    work_size: Sequence[int],
    overlap: Sequence[int],
    orders: Sequence[int] | None = None,
    map_coordinates=ndimage.map_coordinates,
    image_box: bounding_box.BoundingBox | None = None,
    map_box: bounding_box.BoundingBox | None = None,
    out_box: bounding_box.BoundingBox | None = None,
    parallelism: int = 1,
    out_scale: tuple[float, float, float] | tuple[float, float] = (
        1.0,
        1.0,
        1.0,
    ),
) -> list[np.ndarray]:
  """Warps one or more subvolumes sharing the same dense coordinate grid.

  All volumes in `images` must have the same shape and dimensionality matching
  `coord_map`. Dense coordinate upsampling is performed once per work box and
  reused across all input volumes. Any `uint64` volume is treated as
  segmentation data (made contiguous, warped with order 0, and relabeled back
  to its original IDs, overriding the corresponding entry in `orders`).

  Args:
    images: sequence of [z, ] y, x arrays of identical shape to warp
    coord_map: [N, [z,] y, x] coordinate map
    stride: [z,] y, x length in pixels of the image corresponding to a single
      pixel of the coordinate map
    work_size: xy[z] size of the subvolume to warp at a time; use smaller sizes
      to limit RAM usage
    overlap: xy[z] overlap between the subvolumes within which to do warping
    orders: optional per-image interpolation orders passed to `map_coordinates`;
      defaults to 1 for each image (overridden to 0 for `uint64` images)
    map_coordinates: a callable with the signature of ndimage.map_coordinates to
      use for warping
    image_box: bounding box for the image data
    map_box: bounding box for the coordinate map; if specified, image_box has to
      also be defined; if not specified, coord_map's origin is assumed to lie at
      the origin of the input images
    out_box: bounding box for which to generate warped data; if not specified,
      assumed to be the same as image_box
    parallelism: number of threads to use for warping
    out_scale: xy[z] scaling vector out_image_voxel_size /
      source_image_voxel_size

  Returns:
    list of warped arrays corresponding to `images`
  """
  if not images:
    return []
  shape = coord_map.shape[1:]  # ignore xy[z] channel
  dim = len(shape)
  assert dim == len(stride)
  assert dim == len(overlap)
  assert dim == len(work_size)
  ref_shape = images[0].shape
  for img in images:
    if dim != img.ndim:
      raise ValueError(
          f'Dimension mismatch: image: {img.ndim} vs coord map: {dim}'
      )
    if img.shape != ref_shape:
      raise ValueError(
          f'All images must share shape {ref_shape}, got {img.shape}'
      )

  if orders is None:
    orders_list = [1] * len(images)
  else:
    orders_list = list(orders)
    assert len(orders_list) == len(images)

  proc_images: list[np.ndarray] = []
  relabel_info: list[tuple[Any, frozenset[int]] | None] = []
  for idx, img in enumerate(images):
    if img.dtype == np.uint64:
      contig_img, orig_to_low = labels.make_contiguous(img)
      old_uids = frozenset(np.unique(contig_img))
      proc_images.append(contig_img)
      relabel_info.append((orig_to_low, old_uids))
      orders_list[idx] = 0
    else:
      proc_images.append(img)
      relabel_info.append(None)

  src_map = map_utils.to_absolute(coord_map, stride)
  if map_box is not None:
    if image_box is None:
      raise ValueError('image_box has to be specified when map_box is used.')

    # Make map relative to the origin of loaded data (in target space units).
    src_map += (
        map_box.start[:dim] * stride[::-1]
        - image_box.start[:dim] / out_scale[:dim]
    ).reshape(dim, *(1,) * dim)

  # Translate map to source (data) units.
  src_map = src_map * np.asarray(out_scale[:dim]).reshape(dim, *(1,) * dim)

  sub_dim = 0
  image_size_xyz = list(ref_shape[::-1])
  work_size_xyz = list(work_size)
  overlap_xyz = list(overlap)
  if dim == 2:
    work_size_xyz = work_size_xyz + [1]
    overlap_xyz = overlap_xyz + [0]
    image_size_xyz = image_size_xyz + [1]
    sub_dim = 1

  if out_box is not None:
    warped_list = [
        np.zeros(shape=out_box.size[::-1][sub_dim:], dtype=img.dtype)
        for img in proc_images
    ]
  else:
    warped_list = [np.zeros_like(img) for img in proc_images]
    out_box = bounding_box.BoundingBox(start=(0, 0, 0), size=image_size_xyz)

  calc = box_generator.BoxGenerator(
      outer_box=bounding_box.BoundingBox(start=(0, 0, 0), size=out_box.size),  # pyrefly: ignore[bad-argument-type]
      box_size=work_size_xyz,  # pyrefly: ignore[bad-argument-type]
      box_overlap=overlap_xyz,  # pyrefly: ignore[bad-argument-type]
      back_shift_small_boxes=True,
  )

  # Compute the position of map_box relative to the out_box.
  if map_box is not None:
    offset = (map_box.start[:dim] * stride[::-1] - out_box.start[:dim])[::-1]
  else:
    offset = (0,) * dim

  use_fast_upsample = map_coordinates is ndimage.map_coordinates

  def _warp_box(i: int) -> None:
    in_sub_box = calc.generate(i)[1]
    starts = in_sub_box.start[::-1][sub_dim:]
    ends = in_sub_box.end[::-1][sub_dim:]
    if use_fast_upsample:
      src_1d = [
          (np.arange(st, en, dtype=np.float64) - o) / s
          for st, en, s, o in zip(starts, ends, stride, offset)
      ]
      dense_coords = [
          _separable_linear_upsample(eval_coords, src_1d)
          for eval_coords in src_map[::-1]
      ]
    else:
      sel = [np.s_[st:en] for st, en in zip(starts, ends)]
      src_coords = np.mgrid[sel]
      src_coords = [(c - o) / s for c, s, o in zip(src_coords, stride, offset)]
      dense_coords = [
          map_coordinates(eval_coords, src_coords, order=1)
          for eval_coords in src_map[::-1]
      ]

    out_sub_box = calc.index_to_cropped_box(i)
    rel_box = out_sub_box.translate(-in_sub_box.start)  # pyrefly: ignore[bad-argument-type]
    out_sl = out_sub_box.to_slice3d()[sub_dim:]
    rel_sl = rel_box.to_slice3d()[sub_dim:]

    for warped, img, ord_i in zip(warped_list, proc_images, orders_list):
      sub_warped = map_coordinates(img, dense_coords, order=ord_i)
      warped[out_sl] = sub_warped[rel_sl]

  _run_parallel(_warp_box, range(calc.num_boxes), parallelism)

  results: list[np.ndarray] = []
  for warped, rel_info, img in zip(warped_list, relabel_info, images):
    if rel_info is not None:
      orig_to_low, old_uids = rel_info
      warped = _relabel_segmentation(warped, orig_to_low, old_uids)
    results.append(warped.astype(img.dtype, copy=False))
  return results


def ndimage_warp(
    image: np.ndarray,
    coord_map: np.ndarray,
    stride: Sequence[float],
    work_size: Sequence[int],
    overlap: Sequence[int],
    order=1,
    map_coordinates=ndimage.map_coordinates,
    image_box: bounding_box.BoundingBox | None = None,
    map_box: bounding_box.BoundingBox | None = None,
    out_box: bounding_box.BoundingBox | None = None,
    parallelism: int = 1,
    out_scale: tuple[float, float, float] | tuple[float, float] = (
        1.0,
        1.0,
        1.0,
    ),
) -> np.ndarray:
  """Warps a subvolume of data using ndimage.map_coordinates.

  Args:
    image: [z, ] y, x data to warp
    coord_map: [N, [z,] y, x] coordinate map
    stride: [z,] y, x length in pixels of the image corresponding to a single
      pixel of the coordinate map
    work_size: xy[z] size of the subvolume to warp at a time; use smaller sizes
      to limit RAM usage
    overlap: xy[z] overlap between the subvolumes within which to do warping
    order: interpolation order to use (passed to ndimage.map_coordinates)
    map_coordinates: a callable with the signature of ndimage.map_coordinates to
      use for warping
    image_box: bounding box for the image data
    map_box: bounding box for the coordinate map; if specified, image_box has to
      also be defined; if not specified, coord_map's origin is assumed to lie at
      the origin of 'image'
    out_box: bounding box for which to generate warped data; if not specified,
      assumed to be the same as image_box
    parallelism: number of threads to use for warping
    out_scale: xy[z] scaling vector out_image_voxel_size /
      source_image_voxel_size

  Returns:
    warped image
  """
  return ndimage_warp_multi(
      [image],
      coord_map,
      stride,
      work_size,
      overlap,
      orders=[order],
      map_coordinates=map_coordinates,
      image_box=image_box,
      map_box=map_box,
      out_box=out_box,
      parallelism=parallelism,
      out_scale=out_scale,
  )[0]


def render_tiles(
    tiles: dict[tuple[int, int], np.ndarray],
    coord_maps: dict[tuple[int, int], np.ndarray],
    stride: tuple[int, int] = (20, 20),
    margin: int = 50,
    parallelism: int = 1,
    width: int | None = None,
    height: int | None = None,
    use_clahe: bool = False,
    clahe_kwargs=None,
    margin_overrides: (
        dict[tuple[int, int], tuple[int, int, int, int]] | None
    ) = None,
    return_warped_tiles: bool = False,
    tile_masks: dict[tuple[int, int], np.ndarray] | None = None,
    out_box: bounding_box.BoundingBox | None = None,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[tuple[int, int], Any]]
):
  """Warps a collection of tiles into a larger image.

  All values in the 'tiles' and 'positions' maps are assumed to
  have the same shape.

  Args:
    tiles: map from (x, y) tile coordinates to tile image content
    coord_maps: map from (x, y) tile coordinates to coordinate map for the
      corresponding tile; the map is expected to have shape [2,1,my,mx] where mx
      and my are the horizontal/vertical size of the tile, divided by the stride
    stride: stride of the coordinate map in pixels
    margin: number of pixels at the tile edges to exclude from rendering
    parallelism: number of threads used to render the tiles
    width: width of the target image in pixels; inferred from 'tiles' when not
      provided
    height: height of the target image in pixels; inferred from 'tiles' when not
      provided
    use_clahe: whether to apply CLAHE prior to warping
    clahe_kwargs: passed to skimage.exposure.equalize_adapthist
    margin_overrides: optional map from (x, y) tile coordinates to a tuple of
      (top, bottom, left, right) margin sizes in pixels; overrides the global
      default provided in 'margin'.
    return_warped_tiles: whether to return the warped tiles and their relative
      positions.
    tile_masks: map from (x, y) tile coordinates to map arrays (same shape as
      tile images); if present, the elements of the mask evaluating to True
      define the pixels that should be rendered
    out_box: optional xy bounding box defining a target sub-canvas crop to
      render; overrides width and height when specified

  Returns:
    tuple of:
      image with the warped tiles,
      binary array with the same shape as the image;

    'true' pixels in the latter array indicate locations that have been filled
    with tile content during warping; both arrays are (height, width)-shaped

    If the return_warped_tiles argument is True, it also returns a dict from
    tile coords to the warped tiles and their relative positions
    (x0, y0, warped_tile).
  """
  if stride[0] != stride[1]:
    raise NotImplementedError(
        'Currently only equal strides in XY are supported.'
    )
  tile_stride = stride[0]

  any_tile = next(iter(tiles.values()))
  img_yx = any_tile.shape
  image_box = bounding_box.BoundingBox(
      start=(0, 0, 0), size=(img_yx[1], img_yx[0], 1)
  )
  map_yx = next(iter(coord_maps.values())).shape[-2:]
  map_box = bounding_box.BoundingBox(
      start=(0, 0, 0), size=(map_yx[1], map_yx[0], 1)
  )

  warped_image_map = {}

  if out_box is not None:
    canvas_box = bounding_box.BoundingBox(
        start=(int(out_box.start[0]), int(out_box.start[1]), 0),
        size=(int(out_box.size[0]), int(out_box.size[1]), 1),
    )
  else:
    # Infer target image size if necessary.
    if width is None or height is None:
      max_x, max_y = 0, 0
      for x, y in tiles.keys():
        max_x = max(x, max_x)
        max_y = max(y, max_y)

      height, width = img_yx[0] * (max_y + 1), img_yx[1] * (max_x + 1)
    canvas_box = bounding_box.BoundingBox(
        start=(0, 0, 0), size=(int(width), int(height), 1)
    )

  ret = np.zeros(
      (int(canvas_box.size[1]), int(canvas_box.size[0])), dtype=any_tile.dtype
  )
  ret_mask = np.zeros(ret.shape, dtype=bool)

  if clahe_kwargs is None:
    clahe_kwargs = {}

  def _render_tile(item: tuple[tuple[int, int], np.ndarray]) -> None:
    (tile_x, tile_y), coord_map = item
    img = tiles.get((tile_x, tile_y), None)
    if img is None:
      return
    if coord_map.ndim == 3:
      coord_map = coord_map[:, np.newaxis, :, :]

    tg_box = map_utils.outer_box(coord_map, map_box, tile_stride)
    # Add context to avoid rounding issues.
    tg_box = tg_box.adjusted_by(start=(-1, -1, 0), end=(1, 1, 0))

    ox, oy = int(img_yx[1] * tile_x), int(img_yx[0] * tile_y)
    tile_global_box = bounding_box.BoundingBox(
        start=(
            int((tg_box.start[0] + 1) * tile_stride) + ox,
            int((tg_box.start[1] + 1) * tile_stride) + oy,
            0,
        ),
        size=(
            int(tg_box.size[0] * tile_stride),
            int(tg_box.size[1] * tile_stride),
            1,
        ),
    )
    # Position in the target canvas coordinate space.
    clipped_global = tile_global_box.intersection(canvas_box)
    if clipped_global is None:
      return
    canvas_rel = clipped_global.translate(
        (-int(canvas_box.start[0]), -int(canvas_box.start[1]), 0)
    )
    tile_out_box = clipped_global.translate((-ox, -oy, 0))
    x0, y0 = int(canvas_rel.start[0]), int(canvas_rel.start[1])
    canvas_sl = canvas_rel.to_slice3d()[1:]

    inverted_map = map_utils.invert_map(coord_map, map_box, tg_box, tile_stride)
    inverted_map = map_utils.fill_missing(inverted_map, extrapolate=True)
    if np.all(np.isnan(inverted_map)):
      if return_warped_tiles:
        warped_image_map[(tile_x, tile_y)] = (
            x0,
            y0,
            np.zeros(
                (int(tile_out_box.size[1]), int(tile_out_box.size[0])),
                dtype=img.dtype,
            ),
        )
      return

    # Margin removal here is necessary because tiles are sometimes a bit
    # deformed over the first few pixels. Cutting based on actual tile-tile
    # overlaps works, but will leave holes at the corners.
    mask = np.zeros(img.shape, dtype=np.uint8)

    if margin_overrides is not None and (tile_x, tile_y) in margin_overrides:
      mo = margin_overrides[tile_x, tile_y]
      mask[mo[0] : -(mo[1] + 1), mo[2] : -(mo[3] + 1)] = 1
    else:
      mask[margin : -(margin + 1), margin : -(margin + 1)] = 1

    if use_clahe:
      img = (
          skimage.exposure.equalize_adapthist(img, **clahe_kwargs)
          * np.iinfo(img.dtype).max
      ).astype(img.dtype)

    if tile_masks is not None:
      tile_mask = tile_masks.get((tile_x, tile_y), None)
      if tile_mask is not None:
        mask[tile_mask == 0] = 0

    if img.dtype == np.uint64:
      to_warp = np.stack([img, mask.astype(np.uint64)], axis=0)[
          :, np.newaxis, ...
      ]
      warped_img, warped_mask = warp_subvolume(
          to_warp,
          image_box,
          inverted_map,
          tg_box,
          tile_stride,
          out_box=tile_out_box,
      )
      warped_img = warped_img[0, ...]
      warped_mask = warped_mask[0, ...].astype(bool)
    else:
      orig_dtype = img.dtype
      img_cv = _prepare_cv_image(img)
      abs_map, remap_grid = _prepare_cv_remap(
          inverted_map, tg_box, image_box, tile_out_box, tile_stride
      )
      dx_cv, dy_cv = _compute_cv_remap_tables(
          abs_map[:, 0, ...], remap_grid, nninterpolation=False
      )
      warped_img = cv.remap(
          img_cv, dx_cv, dy_cv, interpolation=cv.INTER_LANCZOS4
      ).astype(orig_dtype, copy=False)
      warped_mask = cv.remap(
          mask, dx_cv, dy_cv, interpolation=cv.INTER_LINEAR
      ).astype(bool)

    out = ret[canvas_sl]

    if return_warped_tiles:
      warped_image_map[(tile_x, tile_y)] = x0, y0, warped_img

    ret_mask[canvas_sl][warped_mask] = True

    # If we failed to render any locations in warped_img, do not copy them to
    # the canvas.
    warped_mask &= warped_img > 0
    out[warped_mask] = warped_img[warped_mask]

  _run_parallel(_render_tile, list(coord_maps.items()), parallelism)

  if not return_warped_tiles:
    return ret, ret_mask
  else:
    return ret, ret_mask, warped_image_map


def warp_points(
    points: np.ndarray,
    coord_map: np.ndarray,
    map_box: bounding_box.BoundingBox,
    stride: float,
) -> np.ndarray:
  """Warps a collection of points.

  Preserves the input dtype of point coordinates. Only in-plane warping
  is currently supported. If an integer type is used, transformed coordinates
  are going to be rounded.

  Args:
    points: [n, 3] set of points to map; last dimension stores coordinates in
      XYZ order
    coord_map: [2, z, y, x] coordinate map
    map_box: bounding box of the coordinate map
    stride: stride of the coordinate map in pixels

  Returns:
    [n, 3] array of warped points (same format as 'points')
  """

  points = np.asarray(points)
  assert points.ndim == 2
  assert points.shape[1] == 3
  assert coord_map.shape[0] == 2

  ret = points.copy()
  if points.shape[0] == 0:
    return ret

  hm, wm = coord_map.shape[2], coord_map.shape[3]
  px = points[:, 0].astype(np.float64)
  py = points[:, 1].astype(np.float64)
  z_rel = (points[:, 2] - map_box.start[2]).astype(np.intp)

  if wm >= 2:
    qx = px / stride - map_box.start[0]
    kx = np.clip(np.floor(qx).astype(np.intp), 0, wm - 2)
    tx = qx - kx
    omx = 1.0 - tx
    base_x = px

    def _eval_x(iy):
      return (
          omx * coord_map[:, z_rel, iy, kx]
          + tx * coord_map[:, z_rel, iy, kx + 1]
      )
  else:
    base_x = float(map_box.start[0] * stride)

    def _eval_x(iy):
      return coord_map[:, z_rel, iy, 0]

  if hm >= 2:
    qy = py / stride - map_box.start[1]
    ky = np.clip(np.floor(qy).astype(np.intp), 0, hm - 2)
    ty = qy - ky
    omy = 1.0 - ty
    dxy_rel = omy * _eval_x(ky) + ty * _eval_x(ky + 1)
    base_y = py
  else:
    dxy_rel = _eval_x(0)
    base_y = float(map_box.start[1] * stride)

  dx = (base_x + dxy_rel[0].astype(np.float64, copy=False)).astype(np.float32)
  dy = (base_y + dxy_rel[1].astype(np.float64, copy=False)).astype(np.float32)
  if np.issubdtype(ret.dtype, np.integer):
    dx = np.round(dx)
    dy = np.round(dy)
  ret[:, 0] = dx
  ret[:, 1] = dy
  return ret
