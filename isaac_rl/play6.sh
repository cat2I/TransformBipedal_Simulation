#!/usr/bin/env bash
# =============================================================================
#  play6.sh — xem robot 6 khớp đi bộ (Hip, Knee, Foot — khóa Bub + Twist)
# =============================================================================
#  Cách dùng:
#      ./play6.sh 2026-02-20_23-10-45_final          ← tự tìm checkpoint tốt nhất
#      ./play6.sh 2026-03-24_15-25-04_work 200        ← chỉ định vòng cụ thể
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN="${1:?Thiếu tên run. Ví dụ: ./play6.sh 2026-02-20_23-10-45_final}"
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

echo "🤖 Robot 6 khớp — $(basename "$CKPT")"
exec "$SCRIPT_DIR/run.sh" scripts/rsl_rl/play.py \
    --task Transformer-Walk10DOF6-Direct-v0 \
    --num_envs 1 \
    --checkpoint "$CKPT"
