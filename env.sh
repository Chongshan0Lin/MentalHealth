# source this before any work in this project.  $HOME is FULL (4.5G) - never let caches land there.
export PROJ=/data/cl195/NeurIPS_workshop
export HF_HOME=/data/cl195/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export TORCH_HOME=/data/cl195/cache/torch
export PIP_CACHE_DIR=/data/cl195/cache/pip
export XDG_CACHE_HOME=/data/cl195/cache/xdg
export TMPDIR=/data/cl195/tmp
export CONDA_PKGS_DIRS=/data/cl195/conda/pkgs
export HF_XET_HIGH_PERFORMANCE=1
export SC_ENV=/data/cl195/conda/envs/sc-shortcut
export SC_PY=$SC_ENV/bin/python
