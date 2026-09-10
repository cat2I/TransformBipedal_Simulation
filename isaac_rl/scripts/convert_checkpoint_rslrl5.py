#!/usr/bin/env python3
"""
Chuyển checkpoint rsl_rl cũ (< 4.0) sang định dạng rsl-rl-lib >= 4.0
=====================================================================
Bản cũ lưu MỘT module ``ActorCritic`` chung::

    model_state_dict = {std, actor.0.weight, ..., critic.6.bias}

Bản mới tách thành hai model rời::

    actor_state_dict  = {distribution.std_param, mlp.0.weight, ...}
    critic_state_dict = {mlp.0.weight, ...}

Kiến trúc mạng KHÔNG đổi, mọi shape khớp 1-1, nên đây thuần tuý là đổi tên khoá:

    actor.<i>.<p>   ->  actor_state_dict["mlp.<i>.<p>"]
    critic.<i>.<p>  ->  critic_state_dict["mlp.<i>.<p>"]
    std             ->  actor_state_dict["distribution.std_param"]

``std`` giữ nguyên giá trị: cả hai bản đều lưu độ lệch chuẩn thô (không phải
log-std) — đã kiểm chứng bằng cách so checkpoint mới train với ``init_std=1.0``.

Về optimizer
------------
State của Adam bản cũ gắn với thứ tự tham số của module gộp, không map sang cặp
actor/critic rời được một cách an toàn. Script tạo optimizer TRỐNG (đúng
``param_groups``, ``state`` rỗng). Play không dùng optimizer nên không ảnh hưởng;
train tiếp thì Adam chỉ mất vài chục bước để dựng lại moment.

Dùng
----
    # một file
    python scripts/convert_checkpoint_rslrl5.py logs/.../model_99.pt

    # cả thư mục run
    python scripts/convert_checkpoint_rslrl5.py logs/rsl_rl/transformer_walk/2026-07-27_15-02-49

File cũ được giữ nguyên, bản chuyển đổi ghi vào ``<ten>_rslrl5.pt``.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import torch

# "actor.0.weight" -> ("actor", "0.weight")
_KEY = re.compile(r"^(actor|critic)\.(.+)$")


def convert_state_dict(model_state: dict) -> tuple[dict, dict]:
    """Tách ``model_state_dict`` gộp thành (actor_state_dict, critic_state_dict)."""
    actor: dict = {}
    critic: dict = {}

    for key, value in model_state.items():
        if key == "std":
            actor["distribution.std_param"] = value
            continue
        m = _KEY.match(key)
        if m is None:
            raise ValueError(
                f"Khoá lạ trong model_state_dict: '{key}'. Checkpoint này có thể "
                f"không phải định dạng ActorCritic của rsl_rl < 4.0."
            )
        which, rest = m.group(1), m.group(2)
        (actor if which == "actor" else critic)["mlp." + rest] = value

    if "distribution.std_param" not in actor:
        raise ValueError("Không thấy khoá 'std' — checkpoint thiếu độ lệch chuẩn của policy.")
    if not actor or not critic:
        raise ValueError(f"Tách hỏng: actor={len(actor)} khoá, critic={len(critic)} khoá.")
    return actor, critic


def _empty_optimizer_state(actor: dict, critic: dict) -> dict:
    """Optimizer Adam rỗng nhưng đúng số tham số (actor trước, critic sau)."""
    n = len(actor) + len(critic)
    return {
        "state": {},
        "param_groups": [{
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "params": list(range(n)),
        }],
    }


def convert_file(src: pathlib.Path, force: bool = False) -> pathlib.Path | None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if "actor_state_dict" in ckpt:
        print(f"  BO QUA {src.name}: đã là định dạng mới")
        return None
    if "model_state_dict" not in ckpt:
        print(f"  BO QUA {src.name}: không có 'model_state_dict'")
        return None

    dst = src.with_name(src.stem + "_rslrl5.pt")
    if dst.exists() and not force:
        print(f"  BO QUA {src.name}: '{dst.name}' đã tồn tại (dùng --force để ghi đè)")
        return None

    actor, critic = convert_state_dict(ckpt["model_state_dict"])
    torch.save(
        {
            "actor_state_dict": actor,
            "critic_state_dict": critic,
            "optimizer_state_dict": _empty_optimizer_state(actor, critic),
            "iter": ckpt.get("iter", 0),
            "infos": ckpt.get("infos"),
        },
        dst,
    )
    print(f"  OK  {src.name} -> {dst.name}   (actor {len(actor)} khoá, critic {len(critic)} khoá)")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=pathlib.Path, help="file .pt hoặc thư mục chứa các file .pt")
    ap.add_argument("--force", action="store_true", help="ghi đè bản _rslrl5.pt đã có")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"Không tìm thấy: {args.path}", file=sys.stderr)
        return 1

    if args.path.is_dir():
        files = sorted(p for p in args.path.glob("*.pt") if not p.stem.endswith("_rslrl5"))
        if not files:
            print(f"Không có file .pt nào trong {args.path}", file=sys.stderr)
            return 1
    else:
        files = [args.path]

    print(f"Chuyển đổi {len(files)} file:")
    for f in files:
        try:
            convert_file(f, force=args.force)
        except Exception as exc:  # noqa: BLE001 - báo lỗi từng file, không dừng cả lô
            print(f"  LOI {f.name}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
