#!/usr/bin/env bash
# =============================================================================
#  play10.sh — xem robot 10 khớp đi bộ (Bub, Hip, Twist, Knee, Foot)
# =============================================================================
#  Cách dùng:
#      ./play10.sh 2026-07-27_15-02-49                ← tự tìm checkpoint tốt nhất
#      ./play10.sh 2026-07-23_15-23-03 1499           ← chỉ định vòng cụ thể
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN="${1:?Thiếu tên run. Ví dụ: ./play10.sh 2026-07-27_15-02-49}"
ITER="${2:-}"
LOG_DIR="$SCRIPT_DIR/logs/rsl_rl/transformer_walk/$RUN"

if [[ ! -d "$LOG_DIR" ]]; then
    echo "[LỖI] Không tìm thấy thư mục: $LOG_DIR" >&2
    exit 1
fi

if [[ -n "$ITER" ]]; then
    CKPT="$LOG_DIR/model_${ITER}_rslrl5.pt"
else
    CKPT="$(ls "$LOG_DIR"/model_*_rslrl5.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)"
fi

if [[ ! -f "$CKPT" ]]; then
    echo "[LỖI] Không tìm thấy checkpoint: $CKPT" >&2
    exit 1
fi

echo "🤖 Robot 10 khớp — $(basename "$CKPT")"
exec "$SCRIPT_DIR/run.sh" scripts/rsl_rl/play.py \
    --task Transformer-Walk10DOF-Direct-v0 \
    --num_envs 1 \
    --checkpoint "$CKPT"
