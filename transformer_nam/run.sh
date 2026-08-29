#!/usr/bin/env bash
# =============================================================================
#  run.sh — chạy script Python của project trong môi trường Isaac Sim
# =============================================================================
#  Isaac Sim 6.0.1 + IsaacLab 3.0 ở máy này được cài bằng pip vào conda env
#  'isaacsim', KHÔNG phải bản đóng gói sẵn có thư mục ~/IsaacLab/isaac-sim.
#  Vì vậy không dùng python.sh hay setup_python_env.sh — chỉ cần bật conda env
#  rồi gọi python bình thường.
#
#  Cách dùng:
#      ./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 \
#               --num_envs 512 --headless --max_iterations 350
#
#      ./run.sh scripts/rsl_rl/play.py  --task Transformer-Walk10DOF-Direct-v0 \
#               --num_envs 1 --load_run 2026-07-27_15-02-49
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${ISAAC_CONDA_ENV:-isaaclab30}"

if [[ $# -eq 0 ]]; then
    echo "Dùng: $0 <script.py> [tham số...]" >&2
    echo "  vd: $0 scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 --headless" >&2
    exit 1
fi

# --- bật conda env -----------------------------------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "[LỖI] Không tìm thấy conda ở '$CONDA_BASE'." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    echo "[LỖI] Không có conda env '${CONDA_ENV}'. Các env hiện có:" >&2
    conda env list >&2
    exit 1
fi
conda activate "$CONDA_ENV"

# --- môi trường cho Isaac Sim ------------------------------------------------
export PYTHONPATH="${SCRIPT_DIR}/source${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
# RTX 4050 chỉ có 6GB VRAM, dễ phân mảnh -> cho allocator co giãn
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# tránh PhysX và PyTorch tranh nhau cả 20 thread
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
# Máy dùng đồ hoạ lai, 'prime-select' đang ở chế độ on-demand: desktop chạy trên
# Intel Iris Xe, RTX 4050 chỉ bật khi được gọi tên. Ba biến này đẩy render sang
# 4050 để play có cửa sổ không tụt xuống iGPU (hoặc tệ hơn là llvmpipe phần mềm).
# Khi headless thì vô hại — không có gì để render.
export __NV_PRIME_RENDER_OFFLOAD=1
export __VK_LAYER_NV_optimus=NVIDIA_only
export __GLX_VENDOR_LIBRARY_NAME=nvidia

export QT_QPA_PLATFORM=xcb
export GDK_BACKEND=x11
export SDL_VIDEODRIVER=x11

ulimit -n 65536 2>/dev/null || true

# --- chạy --------------------------------------------------------------------
# isaac-run (do ~/isaac-setup.sh cài) bật CPU hiệu năng cao rồi tự trả về
# powersave lúc thoát. Chưa cài thì chạy thẳng, không sao.
cd "$SCRIPT_DIR"
if command -v isaac-run >/dev/null 2>&1; then
    exec isaac-run python "$@"
else
    exec python "$@"
fi
