#!/usr/bin/env bash
# =============================================================================
#  run_direct.sh — giữ lại cho tương thích, chuyển tiếp sang run.sh
# =============================================================================
#  Bản cũ gọi ${HOME}/IsaacLab/isaac-sim/python.sh — đường dẫn đó chỉ có ở bản
#  Isaac Sim đóng gói sẵn. Máy này cài bằng pip trong conda env 'isaacsim' nên
#  không tồn tại python.sh; mọi thứ giờ đi qua run.sh.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run.sh" "$@"
