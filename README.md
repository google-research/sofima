# SOFIMA

SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment) is a tool
for stitching, aligning and warping large 2d, 3d and 4d microscopy datasets.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is not an officially supported Google product.

# Installation

SOFIMA is implemented purely in Python, and no installation is required. To install
the necessary dependencies, run:

```shell
  pip install -r requirements.txt
```

# Overview

SOFIMA uses optical flow regularized with an elastic mesh to establish
maps between data in different coordinate systems. Both the [flow estimator](flow_field.py)
as well as the [mesh solver](mesh.py) are implemented in [JAX](https://github.com/google/jax)
and will automatically take advantage of GPU acceleration if the hardware if available.

A core data structure used throughout the project is a *coordinate map* stored
as a dense array of relative offsets (see the module docstring in [map_utils.py](map_utils.py)
for details). Among other uses, this is the representation of the estimated flow fields
and the mesh node positions.

# Example usage

 * [electron microscopy tile stitching](https://colab.research.google.com/github/google-research/sofima/blob/main/notebooks/em_stitching.ipynb)

# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
