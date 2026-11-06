#!/usr/bin/env bash
# Local reproduction command. Run with sudo from an interactive terminal.
# Captures one warmed kernel; does not modify driver settings or GPU clocks.
set -euo pipefail
cd /home/lt0649/.codex/worktrees/a6ee/syris_cuda
export PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR=/tmp/syris-profile-admin-cupy
export CUPY_CACHE_SAVE_CUDA_SOURCE=1
export MPLCONFIGDIR=/tmp/syris-profile-admin-mpl
exec /usr/local/cuda/bin/ncu \
    --config-file off --set detailed \
    --section SchedulerStats --section WarpStateStats --section InstructionStats \
    --profile-from-start off --clock-control none --cache-control none \
    --kernel-name 'regex:^project_parallel_normals_kernel$' --launch-count 1 \
    --export /tmp/syris-ncu-armadillo-float32-d1-lineinfo \
    /home/lt0649/Dev/syris_cuda/.venv/bin/python -m benchmarks.profile_cuda_projection \
    --mesh /home/lt0649/.cache/pyvista_3/Armadillo.ply \
    --size 1024 --camera parallel --depths 1 --repeats 1 --line-info \
    --output /tmp/syris-ncu-armadillo-float32-d1-lineinfo.json
