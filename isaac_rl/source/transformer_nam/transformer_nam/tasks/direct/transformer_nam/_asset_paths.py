"""
Giải đường dẫn tới file mô hình trong thư mục ``assets/`` của project
=====================================================================
Trước đây các file config hard-code đường dẫn tuyệt đối kiểu
``/home/tatung/Desktop/Transform_bipedal/transformer_nam/asset/...`` nên chỉ
chạy được trên đúng một máy. Ở đây ta đi ngược lên từ vị trí file này cho tới
khi gặp thư mục ``assets`` → project đặt ở đâu cũng chạy.

Từ 2026-09-09 toàn bộ mô hình robot gom về ``assets/<robot>/usd/``:

    assets/fulltrans/usd/Fulltrans10DOF.usd
    assets/fulltrans/usd/FullForm.usd
    assets/newsimple/usd/NewSimple.usd
    assets/trans3dof/usd/3DOFTrans.usd
    assets/_archive/usd_variants/FullForm111.usd     (biến thể cũ, chỉ khác limit Bub)

Hàm này nhận **tên file** chứ không nhận đường dẫn, nên các config cũ gọi
``asset_path("Fulltrans10DOF.usd")`` không phải sửa gì.

Các file USD dùng tham chiếu tương đối (``./FullForm/FullForm.usd``,
``configuration/Fulltrans_sensor.usd``) nên chỉ cần trả đúng file gốc là mọi
tham chiếu con tự khớp — miễn không tách chúng khỏi thư mục ``usd/`` của mình.
"""

from pathlib import Path

# Thư mục nào tìm trước. ``_archive`` để cuối vì đó là biến thể đã ngừng dùng:
# nếu một tên file tồn tại ở cả hai nơi, bản đang dùng phải thắng.
_SEARCH_ORDER = ("fulltrans", "newsimple", "trans3dof")


def _assets_root() -> Path:
    """Đi ngược lên cây thư mục tìm ``assets/`` của project."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "assets"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Không tìm thấy thư mục 'assets/' ở bất kỳ cấp cha nào của "
        f"{Path(__file__).resolve()}."
    )


def asset_path(name: str) -> str:
    """Trả về đường dẫn tuyệt đối tới file mô hình tên ``name``.

    Args:
        name: Tên file, ví dụ ``"Fulltrans10DOF.usd"``. Không phải đường dẫn.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file nào trùng tên trong ``assets/``.
    """
    root = _assets_root()

    for robot in _SEARCH_ORDER:
        candidate = root / robot / "usd" / name
        if candidate.is_file():
            return str(candidate)

    # Chưa thấy ở các robot đang dùng → quét rộng (gồm cả _archive/).
    matches = sorted(p for p in root.rglob(name) if p.is_file())
    if matches:
        return str(matches[0])

    raise FileNotFoundError(
        f"Không tìm thấy '{name}' trong '{root}'. Kiểm tra file có nằm trong "
        f"assets/<robot>/usd/ không."
    )
