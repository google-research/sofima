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

from concurrent import futures
from typing import Dict, Optional, Sequence, Tuple
from connectomics.common import bounding_box
from connectomics.common import box_generator
from connectomics.segmentation import labels
# pylint:disable=g-import-not-at-top
try:
  from cvx2 import latest as cvx2
except ImportError:
  import cv2 as cvx2  # pytype:disable=import-error
import numpy as np
from scipy import interpolate
from scipy import ndimage
import skimage.exposure
from sofima import map_utils
# pylint:enable=g-import-not-at-top


def _cvx2_interpolation(inter_scheme: str):
  inter_map = {
      'nearest': cvx2.INTER_NEAREST,
      'linear': cvx2.INTER_LINEAR,
      'cubic': cvx2.INTER_CUBIC,
      'lanczos': cvx2.INTER_LANCZOS4
  }
  return inter_map[inter_scheme]


def _relabel_segmentation(data, orig_to_low, old_uids):
  new_uids = frozenset(np.unique(data.astype(np.uint64)))

  # No new IDs are introduced by the warping.
  diff_ids = (new_uids - old_uids) - {0}
  assert not diff_ids, f'Found unexpected new IDs: {diff_ids}'

  orig_ids, low_ids = zip(*orig_to_low)
  return labels.relabel(
      data.astype(np.uint64), np.array(low_ids, dtype=np.uint64),
      np.array(orig_ids, dtype=np.uint64))


def warp_subvolume(image: np.ndarray,
                   image_box: bounding_box.BoundingBoxBase,
                   coord_map: np.ndarray,
                   map_box: bounding_box.BoundingBoxBase,
                   stride: float,
                   out_box: bounding_box.BoundingBoxBase,
                   interpolation: Optional[str] = None,
                   offset: float = 0.) -> np.ndarray:
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
    offset: (deprecated do not use) non-zero values necessary to reproduce some
      old renders

  Returns:
    warped image covering 'out_box'
  """

  # Segmentation warping.
  if image.dtype == np.uint64:
    interpolation = cvx2.INTER_NEAREST
    image, orig_to_low = labels.make_contiguous(image)
    assert np.max(image) < 2**31
    assert np.min(image) >= 0
    image = image.astype(np.int32)
    old_uids = frozenset(np.unique(image))
  # Image warping.
  else:
    orig_to_low = None
    if interpolation is None:
      interpolation = cvx2.INTER_LANCZOS4
    elif isinstance(interpolation, str):
      interpolation = _cvx2_interpolation(interpolation)

    orig_dtype = image.dtype
    if image.dtype == np.uint32:
      if image.max() >= 2**16:
        raise ValueError(
            'Image warping supported up to uint16 only. For segmentation data, '
            'use uint64.')
      image = image.astype(np.uint16)

  skipped_sections = frozenset(
      np.where(np.all(np.isnan(coord_map), axis=(0, 2, 3)))[0])

  # Convert values within the coordinate map so that they are
  # within the local coordinate system of 'image'.
  abs_map = map_utils.to_absolute(coord_map, stride)
  abs_map += (map_box.start[:2] * stride - image_box.start[:2] +
              offset).reshape(2, 1, 1, 1)

  # Coordinates of the map nodes within the local coordinate
  # system of 'out_box'.
  map_y, map_x = np.ogrid[:coord_map.shape[2], :coord_map.shape[3]]
  map_y = (map_y + map_box.start[1]) * stride - out_box.start[1] + offset
  map_x = (map_x + map_box.start[0]) * stride - out_box.start[0] + offset
  map_points = (map_y.ravel(), map_x.ravel())

  warped = np.zeros(
      shape=[image.shape[0]] + list(out_box.size[::-1]), dtype=image.dtype)
  out_y, out_x = np.mgrid[:out_box.size[1], :out_box.size[0]]

  try:
    maptype = cvx2.CVX_16SC2
  except AttributeError:
    maptype = cvx2.CV_16SC2

  for z in range(0, image.shape[1]):
    if z in skipped_sections:
      continue

    dense_x = interpolate.RegularGridInterpolator(
        map_points, abs_map[0, z, ...], bounds_error=False, fill_value=None)
    dense_y = interpolate.RegularGridInterpolator(
        map_points, abs_map[1, z, ...], bounds_error=False, fill_value=None)

    # dxy: [0 .. out_box.size] -> [coord within image]
    dx = dense_x((out_y, out_x)).astype(np.float32)
    dy = dense_y((out_y, out_x)).astype(np.float32)

    dx, dy = cvx2.convertMaps(
        dx,
        dy,
        dstmap1type=maptype,
        nninterpolation=(interpolation == cvx2.INTER_NEAREST))

    for c in range(0, image.shape[0]):
      warped[c, z, ...] = cvx2.remap(
          image[c, z, ...], dx, dy, interpolation=interpolation)

  # Map IDs back to the original space, which might be beyond the range of
  # int32.
  if orig_to_low is not None:
    warped = _relabel_segmentation(warped, orig_to_low, old_uids)
  else:
    warped = warped.astype(orig_dtype)

  return warped


def ndimage_warp(
    image: np.ndarray,
    coord_map: np.ndarray,
    stride: Sequence[float],
    work_size: Sequence[int],
    overlap: Sequence[int],
    order=1,
    map_coordinates=ndimage.map_coordinates,
    image_box: Optional[bounding_box.BoundingBoxBase] = None,
    map_box: Optional[bounding_box.BoundingBoxBase] = None,
    out_box: Optional[bounding_box.BoundingBoxBase] = None) -> np.ndarray:
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

  Returns:
    warped image
  """
  shape = coord_map.shape[1:]  # ignore xy[z] channel
  dim = len(shape)
  assert dim == len(stride)
  assert dim == len(overlap)
  assert dim == len(work_size)
  assert dim == image.ndim

  orig_to_low = None
  if image.dtype == np.uint64:
    image, orig_to_low = labels.make_contiguous(image)
    old_uids = frozenset(np.unique(image))
    order = 0

  src_map = map_utils.to_absolute(coord_map, stride)
  if map_box is not None:
    if image_box is None:
      raise ValueError('image_box has to be specified when map_box is used.')

    src_map += (map_box.start[:dim] * stride[::-1] -
                image_box.start[:dim]).reshape(dim, 1, 1, 1)

  sub_dim = 0
  image_size_xyz = image.shape[::-1]
  if dim == 2:
    work_size = list(work_size) + [1]
    overlap = list(overlap) + [0]
    image_size_xyz = list(image_size_xyz) + [1]
    sub_dim = 1

  if out_box is not None:
    warped = np.zeros(shape=out_box.size[::-1], dtype=image.dtype)
  else:
    warped = np.zeros_like(image)
    out_box = bounding_box.BoundingBox(start=(0, 0, 0), size=image_size_xyz)

  calc = box_generator.BoxGenerator(
      outer_box=bounding_box.BoundingBox(start=(0, 0, 0), size=out_box.size),
      box_size=work_size,
      box_overlap=overlap,
      back_shift_small_boxes=True)

  if map_box is not None:
    assert out_box is not None
    offset = (map_box.start * stride[::-1] - out_box.start)[::-1]
  else:
    offset = (0, 0, 0)

  for i in range(calc.num_boxes):
    in_sub_box = calc.generate(i)[1]
    sel = [
        np.s_[start:end] for start, end in zip(in_sub_box.start[::-1][sub_dim:],
                                               in_sub_box.end[::-1][sub_dim:])
    ]
    src_coords = np.mgrid[sel]
    src_coords = [(c - o) / s for c, s, o in zip(src_coords, stride, offset)]
    dense_coords = [
        map_coordinates(eval_coords, src_coords, order=1)
        for eval_coords in src_map[::-1]
    ]

    out_sub_box = calc.index_to_cropped_box(i)

    # Warp image data for the current subvolume.
    sub_warped = map_coordinates(image, dense_coords, order=order)
    rel_box = out_sub_box.translate(-in_sub_box.start)

    warped[out_sub_box.to_slice3d()[sub_dim:]] = sub_warped[
        rel_box.to_slice3d()[sub_dim:]]

  if orig_to_low is not None:
    warped = _relabel_segmentation(warped, orig_to_low, old_uids)

  return warped.astype(image.dtype)


def render_tiles(
    tiles: Dict[Tuple[int, int], np.ndarray],
    coord_maps: Dict[Tuple[int, int], np.ndarray],
    stride: Tuple[int, int] = (20, 20),
    margin: int = 50,
    parallelism: int = 1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    use_clahe: bool = False,
    clahe_kwargs: ... = None,
    margin_overrides: Optional[Dict[Tuple[int, int], Tuple[int, int, int,
                                                           int]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
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

  Returns:
    tuple of:
      image with the warped tiles,
      binary array with the same shape as the image;

    'true' pixels in the latter array indicate locations that have been filled
    with tile content during warping; both arrays are (height, width)-shaped
  """
  if stride[0] != stride[1]:
    raise NotImplementedError(
        'Currently only equal strides in XY are supported.')

  any_tile = next(iter(tiles.values()))
  img_yx = any_tile.shape
  image_box = bounding_box.BoundingBox(
      start=(0, 0, 0), size=(img_yx[1], img_yx[0], 1))
  map_yx = next(iter(coord_maps.values())).shape[-2:]
  map_box = bounding_box.BoundingBox(
      start=(0, 0, 0), size=(map_yx[1], map_yx[0], 1))

  # Infer target image size if necessary.
  if width is None or height is None:
    max_x, max_y = 0, 0
    for x, y in tiles.keys():
      max_x = max(x, max_x)
      max_y = max(y, max_y)

    height, width = img_yx[0] * (max_y + 1), img_yx[1] * (max_x + 1)

  ret = np.zeros((height, width), dtype=any_tile.dtype)
  ret_mask = np.zeros((height, width), dtype=bool)

  if clahe_kwargs is None:
    clahe_kwargs = {}

  def _render_tile(tile_x, tile_y, coord_map):
    img = tiles.get((tile_x, tile_y), None)
    if img is None:
      return

    tg_box = map_utils.outer_box(coord_map, map_box, stride[0])
    # Add context to avoid rounding issues.
    tg_box = tg_box.adjusted_by(start=(-1, -1, 0), end=(1, 1, 0))
    inverted_map = map_utils.invert_map(coord_map, map_box, tg_box, stride[0])
    inverted_map = map_utils.fill_missing(inverted_map, extrapolate=True)

    # Margin removal here is necessary because tiles are sometimes a bit
    # deformed over the first few pixels. Cutting based on actual tile-tile
    # overlaps works, but will leave holes at the corners.
    mask = np.zeros_like(img)

    if margin_overrides is not None and (tile_x, tile_y) in margin_overrides:
      mo = margin_overrides[tile_x, tile_y]
      mask[mo[0]:-(mo[1] + 1), mo[2]:-(mo[3] + 1)] = 1
    else:
      mask[margin:-(margin + 1), margin:-(margin + 1)] = 1

    if use_clahe:
      img = (skimage.exposure.equalize_adapthist(img, **clahe_kwargs) *
             np.iinfo(img.dtype).max).astype(img.dtype)

    to_warp = np.concatenate(
        [img[np.newaxis, np.newaxis, ...], mask[np.newaxis, np.newaxis, ...]],
        axis=0)

    out_box = image_box.translate(((tg_box.start[0] + 1) * stride[1],
                                   (tg_box.start[1] + 1) * stride[0], 0))
    out_box = bounding_box.BoundingBox(
        start=out_box.start,
        size=(tg_box.size[0] * stride[1], tg_box.size[1] * stride[0], 1))

    warped_img, warped_mask = warp_subvolume(
        to_warp, image_box, inverted_map, tg_box, stride[0], out_box=out_box)

    warped_img = warped_img[0, ...]
    warped_mask = warped_mask[0, ...].astype(bool)

    # Position in the global coordinate space is relative to the default tile
    # position.
    y0 = img_yx[0] * tile_y + out_box.start[1]
    x0 = img_yx[1] * tile_x + out_box.start[0]

    # Trim warped content if necessary.
    if x0 < 0:
      warped_img = warped_img[:, -x0:]
      warped_mask = warped_mask[:, -x0:]
      x0 = 0

    if y0 < 0:
      warped_img = warped_img[-y0:, :]
      warped_mask = warped_mask[-y0:, :]
      y0 = 0

    out = ret[y0:y0 + warped_img.shape[0], x0:x0 + warped_img.shape[1]]
    os = out.shape
    warped_mask = warped_mask[:os[0], :os[1]]
    warped_img = warped_img[:os[0], :os[1]]

    ret_mask[y0:y0 + warped_img.shape[0],
             x0:x0 + warped_img.shape[1]][warped_mask] = True

    # If we failed to render any locations in warped_img, do not copy them to
    # the canvas.
    warped_mask &= warped_img > 0
    out[warped_mask] = warped_img[warped_mask]

  if parallelism > 1:
    fs = set()
    with futures.ThreadPoolExecutor(max_workers=parallelism) as exc:
      for (x, y), coord_map in coord_maps.items():
        fs.add(
            exc.submit(_render_tile, tile_x=x, tile_y=y, coord_map=coord_map))

      for f in futures.as_completed(fs):
        f.result()
  else:
    for (x, y), coord_map in coord_maps.items():
      _render_tile(tile_x=x, tile_y=y, coord_map=coord_map)

  return ret, ret_mask
