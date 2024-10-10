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

"""Tests for the client utility module."""

from absl.testing import absltest
from sofima.processor import client_utils


class ClientTest(absltest.TestCase):

  def test_get_block_id(self):
    # blocks are: 0..49, 50..99, 100..149, 150..199, 200..
    fwd_starts = [0, 50, 100, 150, 200]
    self.assertEqual(client_utils.get_block_id(10, fwd_starts, False), 1)
    self.assertEqual(client_utils.get_block_id(0, fwd_starts, False), 1)
    self.assertEqual(client_utils.get_block_id(49, fwd_starts, False), 1)
    self.assertEqual(client_utils.get_block_id(50, fwd_starts, False), 2)

    # blocks are: 0..50, 51..100, 101..150, 151..200, 200..
    bwd_starts = [50, 100, 150, 200]
    self.assertEqual(client_utils.get_block_id(10, bwd_starts, True), 0)
    self.assertEqual(client_utils.get_block_id(0, bwd_starts, True), 0)
    self.assertEqual(client_utils.get_block_id(50, bwd_starts, True), 0)
    self.assertEqual(client_utils.get_block_id(51, bwd_starts, True), 1)
    self.assertEqual(client_utils.get_block_id(100, bwd_starts, True), 1)


if __name__ == "__main__":
  absltest.main()
