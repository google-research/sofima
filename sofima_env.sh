#!/bin/bash
conda create --name py311 -c conda-forge python=3.11 -y
conda run -n py311 pip install git+https://github.com/google-research/sofima
conda run -n py311 pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda run -n py311 pip install tensorstore