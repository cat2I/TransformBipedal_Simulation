"""
Giải đường dẫn tới thư mục ``asset/`` của project
==================================================
Trước đây các file config hard-code đường dẫn tuyệt đối kiểu
``/home/tatung/Desktop/Transform_bipedal/transformer_nam/asset/...`` nên chỉ
chạy được trên đúng một máy. Ở đây ta đi ngược lên từ vị trí file này cho tới
khi gặp thư mục ``asset`` chứa file cần tìm → project đặt ở đâu cũng chạy.

Các file USD dùng sublayer tương đối (``configuration/Fulltrans_sensor.usd``)
nên chỉ cần trỏ đúng thư mục ``asset/`` là mọi tham chiếu con tự khớp.
"""

from pathlib import Path


def asset_path(name: str) -> str:
    """Trả về đường dẫn tuyệt đối tới ``asset/<name>``.

    Args:
        name: Tên file trong thư mục asset, ví dụ ``"Fulltrans10DOF.usd"``.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file ở bất kỳ cấp thư mục cha nào.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "asset" / name
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Không tìm thấy 'asset/{name}'. Kiểm tra file có nằm trong thư mục "
        f"asset/ của project không (bắt đầu tìm từ {Path(__file__).resolve()})."
    )
