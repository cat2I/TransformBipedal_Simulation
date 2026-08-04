#!/usr/bin/env bash
# =============================================================================
#  isaaclab.sh — giữ lại cho tương thích, chuyển tiếp sang run.sh
# =============================================================================
#  Bản cũ sai ba chỗ so với máy hiện tại:
#    * conda activate isaaclab      -> env đúng tên là 'isaacsim'
#    * ${HOME}/IsaacLab/isaac-sim   -> không tồn tại (cài bằng pip, không phải
#                                      bản đóng gói sẵn)
#    * .../python3.11/site-packages -> env hiện tại chạy Python 3.12
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run.sh" "$@"
