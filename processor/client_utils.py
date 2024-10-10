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

"""Utility functions for the mesh relxation client."""

import bisect
from collections.abc import Sequence


def get_block_id(z: int, starts: Sequence[int], backward: bool) -> int:
  """Returns the block number to which the section at 'z' belongs."""
  if backward:
    return bisect.bisect_left(starts, z)
  else:
    return bisect.bisect_right(starts, z)
