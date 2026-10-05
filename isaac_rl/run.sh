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
#      ./run.sh scripts/rsl_rl/train.py --task Official-Walk-v0 \
#               --num_envs 512 --headless --max_iterations 350
#
#      ./run.sh scripts/rsl_rl/play.py  --task Official-Walk-v0 \
#               --num_envs 1 --load_run 2026-07-27_15-02-49
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tên conda env khác nhau giữa các máy trong nhóm: máy này là 'isaacsim', máy
# Vinh là 'isaaclab30'. Trước đây dòng này hard-code một tên nên máy còn lại
# luôn chết ở bước kiểm tra env, TRƯỚC cả khi đọc tham số — nên mọi cờ như
# --load_run trông như "không dùng được". Giờ dò trong danh sách ứng viên và
# lấy env đầu tiên có thật; đặt ISAAC_CONDA_ENV để ép một tên cụ thể.
ISAAC_ENV_CANDIDATES=(isaacsim isaaclab30 isaaclab)

if [[ $# -eq 0 ]]; then
    echo "Dùng: $0 <script.py> [tham số...]" >&2
    echo "  vd: $0 scripts/rsl_rl/train.py --task Official-Walk-v0 --headless" >&2
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

env_exists() { conda env list | grep -qE "^${1}[[:space:]]"; }

if [[ -n "${ISAAC_CONDA_ENV:-}" ]]; then
    # Người dùng ép tên cụ thể -> tôn trọng, sai thì báo luôn chứ không dò tiếp.
    CONDA_ENV="$ISAAC_CONDA_ENV"
    if ! env_exists "$CONDA_ENV"; then
        echo "[LỖI] ISAAC_CONDA_ENV='${CONDA_ENV}' nhưng env đó không tồn tại." >&2
        conda env list >&2
        exit 1
    fi
else
    CONDA_ENV=""
    for candidate in "${ISAAC_ENV_CANDIDATES[@]}"; do
        if env_exists "$candidate"; then CONDA_ENV="$candidate"; break; fi
    done
    if [[ -z "$CONDA_ENV" ]]; then
        echo "[LỖI] Không tìm thấy conda env nào trong: ${ISAAC_ENV_CANDIDATES[*]}" >&2
        echo "       Đặt ISAAC_CONDA_ENV=<tên> nếu env của bạn tên khác. Env hiện có:" >&2
        conda env list >&2
        exit 1
    fi
fi
conda activate "$CONDA_ENV"

# --- môi trường cho Isaac Sim ------------------------------------------------
# Trỏ vào gốc isaac_rl/ (package `bipedal` nằm ngay đó). Trước đây trỏ source/,
# nhưng thư mục đó chỉ là vỏ của IsaacLab extension template — đã bỏ 2026-10.
# Nhờ dòng này, `import bipedal` chạy được KỂ CẢ KHI chưa `pip install -e`.
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
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
