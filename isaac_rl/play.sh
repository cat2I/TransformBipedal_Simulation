#!/usr/bin/env bash
# =============================================================================
#  play.sh — phát lại một checkpoint đã train
# =============================================================================
#  Thay cho play6.sh / play10.sh cũ (hai file đó khoá cứng vào log dir của
#  robot cũ nên thêm robot mới là phải viết thêm script).
#
#  CÁCH DÙNG
#      ./play.sh                          liệt kê robot đang có
#      ./play.sh <robot>                  liệt kê các run của robot đó
#      ./play.sh <robot> <run>            phát checkpoint MỚI NHẤT của run
#      ./play.sh <robot> <run> <iter>     phát đúng iteration đó
#
#      Tham số thừa chuyển thẳng cho play.py:
#      ./play.sh official <run> 500 --num_envs 4
#
#  VÍ DỤ — phát lại policy đứng sau 0319.gif (robot thật đã đi bộ):
#      ./play.sh newsimple 2026-03-19_13-18-11_work 349
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── BẢNG ĐĂNG KÝ ROBOT ──────────────────────────────────────────────────────
# Thêm robot mới = thêm 1 dòng vào mỗi mảng dưới đây + 1 dòng vào ORDER.

ORDER=(official newsimple)

declare -A TASK=(
    [official]="Official-Walk-v0"
    [newsimple]="NewSimple-Walk-v0"
)

# Thư mục log dưới logs/, có thể nhiều cái cho một robot (ngăn bởi dấu cách).
# newsimple có HAI: run mới vào logs/newsimple/, còn model_349 nằm trong
# logs/old/ — 96 run của nhiều task trộn lẫn từ thời cả nhóm cùng dùng chung
# experiment_name "transformer_walk", không phân biệt được bằng tên nên không
# tách ra được. Script tìm lần lượt theo thứ tự liệt kê.
declare -A LOGDIRS=(
    [official]="officialdesign"
    [newsimple]="newsimple old"
)

declare -A ASSET=(
    [official]="assets/officialdesign  — 13 link, 10 DOF, obs 60 / act 10"
    [newsimple]="assets/newsimple      —  6 DOF, obs 44 / act  6"
)

declare -A NOTE=(
    [official]="robot mới, đang train (mới chỉ có smoke test 2 iteration)"
    [newsimple]="model_349 — policy DUY NHẤT đã đi được trên robot thật (0319.gif)"
)

# Cờ hydra ép env về đúng điều kiện lúc train. newsimple cần nguyên bộ này, nếu
# không policy cũ sẽ chạy trong môi trường có nhiễu/bias khác lúc nó học.
declare -A EXTRA=(
    [newsimple]="env.domain_rand=False env.imu_fixed_bias=[0.0,0.0,0.0] env.imu_noise_std.orientation=0.015 env.imu_noise_std.angular_velocity=0.01 env.imu_drift_rate=0.0"
)

# ── Hàm phụ ─────────────────────────────────────────────────────────────────

die() { echo "[LỖI] $*" >&2; exit 1; }

list_robots() {
    echo
    echo "  Dùng:  ./play.sh <robot> <run> [iter] [cờ thêm cho play.py...]"
    echo
    printf "  %-13s %-28s %s\n" "ROBOT" "LOG DIR" "GHI CHÚ"
    printf "  %-13s %-28s %s\n" "-----" "-------" "-------"
    for r in "${ORDER[@]}"; do
        local n=0 d
        for d in ${LOGDIRS[$r]}; do
            [[ -d "logs/$d" ]] && n=$((n + $(find "logs/$d" -maxdepth 1 -mindepth 1 -type d | wc -l)))
        done
        printf "  %-13s %-28s %s\n" "$r" "${LOGDIRS[$r]// //} ($n run)" "${NOTE[$r]}"
    done
    echo
    echo "  Gõ ./play.sh <robot> để xem danh sách run."
    echo
}

list_runs() {
    local robot=$1
    echo
    echo "  robot : $robot"
    echo "  task  : ${TASK[$robot]}"
    echo "  asset : ${ASSET[$robot]}"
    echo "  log   : ${LOGDIRS[$robot]// /, } (dưới logs/)"
    [[ -n "${EXTRA[$robot]:-}" ]] && echo "  cờ env: ${EXTRA[$robot]}"
    echo
    echo "  RUN                                ITER CÓ SẴN"
    echo "  ---                                -----------"
    local d run iters found=0
    for d in ${LOGDIRS[$robot]}; do
        [[ -d "logs/$d" ]] || continue
        while IFS= read -r run; do
            found=1
            iters=$(find "logs/$d/$run" -maxdepth 1 -name 'model_*.pt' 2>/dev/null |
                    sed -nE 's/.*model_([0-9]+)(_rslrl5)?\.pt$/\1/p' | sort -un | tr '\n' ' ')
            [[ -z $iters ]] && iters="(không có checkpoint)"
            printf "  %-34s %s\n" "$run" "$iters"
        done < <(find "logs/$d" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
    done
    (( found )) || die "Robot '$robot' chưa train lần nào."
    echo
}

# Chọn file checkpoint. Ưu tiên bản *_rslrl5.pt: checkpoint train bằng rsl_rl cũ
# (<4.0) lưu MỘT module ActorCritic gộp, bản mới đòi actor/critic tách rời.
# convert_checkpoint_rslrl5.py sinh ra bản _rslrl5. Robot mới train thẳng bằng
# rsl-rl mới nên chỉ có model_N.pt, không có bản chuyển.
pick_checkpoint() {
    local dir=$1 iter=${2:-}
    if [[ -z $iter ]]; then
        iter=$(find "$dir" -maxdepth 1 -name 'model_*.pt' |
               sed -nE 's/.*model_([0-9]+)(_rslrl5)?\.pt$/\1/p' | sort -n | tail -1)
        [[ -n $iter ]] || die "Không có file model_*.pt nào trong '$dir'."
    fi
    local f
    for f in "$dir/model_${iter}_rslrl5.pt" "$dir/model_${iter}.pt"; do
        [[ -f $f ]] && { printf '%s' "$f"; return 0; }
    done
    die "Không tìm thấy model_${iter}.pt (hoặc bản _rslrl5) trong '$dir'."
}

# ── Thân chính ──────────────────────────────────────────────────────────────

[[ $# -eq 0 ]] && { list_robots; exit 0; }

ROBOT=$1; shift
[[ -v TASK[$ROBOT] ]] || { echo "[LỖI] Không có robot '$ROBOT'." >&2; list_robots; exit 1; }

[[ $# -eq 0 ]] && { list_runs "$ROBOT"; exit 0; }

RUN=$1; shift
RUN_DIR=""
for d in ${LOGDIRS[$ROBOT]}; do
    [[ -d "logs/$d/$RUN" ]] && { RUN_DIR="logs/$d/$RUN"; break; }
done
[[ -n $RUN_DIR ]] || { echo "[LỖI] Không có run '$RUN'." >&2; list_runs "$ROBOT"; exit 1; }

ITER=""
if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then ITER=$1; shift; fi

CKPT=$(pick_checkpoint "$RUN_DIR" "$ITER")

read -ra EXTRA_ARGS <<< "${EXTRA[$ROBOT]:-}"

# Mặc định 1 env cho dễ nhìn, nhưng nếu người dùng đã tự truyền --num_envs thì
# đừng thêm nữa (argparse lấy cái cuối nên vẫn chạy, chỉ là lệnh in ra bị lặp).
NUM_ENVS=(--num_envs 1)
for a in "$@"; do [[ $a == --num_envs* ]] && NUM_ENVS=() && break; done

echo
echo "  robot      : $ROBOT"
echo "  task       : ${TASK[$ROBOT]}"
echo "  checkpoint : $CKPT"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  cờ env     : ${EXTRA_ARGS[*]}"
echo

exec ./run.sh scripts/rsl_rl/play.py \
    --task "${TASK[$ROBOT]}" \
    "${NUM_ENVS[@]}" \
    --checkpoint "$PWD/$CKPT" \
    "${EXTRA_ARGS[@]}" "$@"
