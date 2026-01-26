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

from absl.testing import absltest
from connectomics.common import bounding_box
from connectomics.volume import subvolume
import numpy as np
from sofima.processor import flow


class MockVolume:

  def __init__(self, data):
    self._data = data  # CZYX

  def clip_box_to_volume(self, box):
    vol_box = bounding_box.BoundingBox(start=(0, 0, 0), size=self.volume_size)
    return box.intersection(vol_box)

  @property
  def asarray(self):
    return self._data

  @property
  def volume_size(self):
    # XYZ
    return (self._data.shape[3], self._data.shape[2], self._data.shape[1])

  def __getitem__(self, key):
    return self._data[key]


class TestEstimateMissingFlow(flow.EstimateMissingFlow):

  def __init__(self, config, image_vol):
    super().__init__(config)
    self.image_vol = image_vol

  def _open_volume(self, path):
    return self.image_vol


class EstimateMissingFlowTest(absltest.TestCase):

  def test_process(self):
    config = flow.EstimateMissingFlow.Config(
        patch_size=16,
        stride=16,
        delta_z=1,
        max_delta_z=2,
        max_attempts=1,
        mask_configs=None,
        mask_only_for_patch_selection=False,
        selection_mask_configs=None,
        min_peak_sharpness=0.0,
        min_peak_ratio=0.0,
        max_magnitude=0,
        batch_size=10,  # Must be > 0 for batch processing
        image_volinfo="dummy_path",
        image_cache_bytes=0,
        mask_cache_bytes=0,
        search_radius=16,
    )

    # Larger volume to avoid boundary clipping with required context size
    vol_shape = (1, 10, 128, 128)
    vol_data = np.random.rand(*vol_shape).astype(np.float32)

    # Create a synthetic shift between z=3 and z=5.
    dx, dy = 2, 3
    prev_slice = vol_data[0, 3, :, :]
    shifted_slice = np.zeros_like(prev_slice)
    shifted_slice[dy:, dx:] = prev_slice[:-dy, :-dx]
    shifted_slice[:dy, :] = np.random.rand(dy, 128)
    shifted_slice[:, :dx] = np.random.rand(128, dx)

    vol_data[0, 5, :, :] = shifted_slice

    mock_vol = MockVolume(vol_data)
    processor = TestEstimateMissingFlow(config, mock_vol)

    # Start at 2,2,5 (flow coords) corresponds to 32,32,5 (image coords).
    box = bounding_box.BoundingBox((2, 2, 5), (2, 2, 1))

    # No pre-existing flow data.
    input_data = np.full((2, 1, 2, 2), np.nan, dtype=np.float32)
    subvol = subvolume.Subvolume(input_data, box)

    result_subvol = processor.process(subvol)

    self.assertEqual(result_subvol.data.shape, (3, 1, 2, 2))
    self.assertFalse(
        np.any(np.isnan(result_subvol.data)), "Result contains NaNs"
    )

    np.testing.assert_allclose(
        result_subvol.data[2, ...], 2, err_msg="delta_z incorrect"
    )
    np.testing.assert_allclose(
        result_subvol.data[0, 0, 0, 0],
        -dx,
        atol=0.5,
        err_msg="Flow X incorrect",
    )
    np.testing.assert_allclose(
        result_subvol.data[1, 0, 0, 0],
        -dy,
        atol=0.5,
        err_msg="Flow Y incorrect",
    )

  def test_process_clipped_context(self):
    config = flow.EstimateMissingFlow.Config(
        patch_size=16,
        stride=16,
        delta_z=1,
        max_delta_z=5,  # Large lookback
        max_attempts=1,
        mask_configs=None,
        mask_only_for_patch_selection=False,
        selection_mask_configs=None,
        min_peak_sharpness=0.0,
        min_peak_ratio=0.0,
        max_magnitude=0,
        batch_size=10,
        image_volinfo="dummy_path",
        image_cache_bytes=0,
        mask_cache_bytes=0,
        search_radius=16,
    )

    vol_shape = (1, 10, 128, 128)
    vol_data = np.random.rand(*vol_shape).astype(np.float32)

    mock_vol = MockVolume(vol_data)
    processor = TestEstimateMissingFlow(config, mock_vol)

    box = bounding_box.BoundingBox(start=(2, 2, 1), size=(2, 2, 1))

    # No pre-existing flow data.
    input_data = np.full((2, 1, 2, 2), np.nan, dtype=np.float32)
    subvol = subvolume.Subvolume(input_data, box)

    result_subvol = processor.process(subvol)

    self.assertEqual(result_subvol.data.shape, (3, 1, 2, 2))

    # Result should be NaNs because z=1 only has z=0 as valid prev.
    # delta_z=1 (matching z=0) was not calculated (assumed missing).
    # delta_z=2,3,4,5 look at z < 0, which is out of bounds.
    self.assertTrue(
        np.all(np.isnan(result_subvol.data[0, ...])), "Result X should be NaN"
    )
    self.assertTrue(
        np.all(np.isnan(result_subvol.data[1, ...])), "Result Y should be NaN"
    )
    # Channel 2 is initialized to delta_z (1).
    self.assertEqual(result_subvol.data[2, 0, 0, 0], 1)


if __name__ == "__main__":
  absltest.main()
