# _archive — không dùng nữa

Giữ lại để đối chiếu. Không có gì ở đây được code nạp trong lúc train.

## `usd_variants/` — biến thể chỉ khác giới hạn khớp

| File | Bubleft/right | 7 khớp còn lại |
|---|---|---|
| `../fulltrans/usd/FullForm.usd` | ±10.03° | *(bản gốc)* |
| `FullForm111.usd` | ±90° | giống hệt tới từng chữ số thập phân |
| `FullForm123.usd` | ±60° | giống hệt |

Ba file khác nhau **đúng 2 con số**. Chúng sinh ra từ việc mở Isaac Sim GUI, sửa giới hạn,
rồi Save As — nên tên là `111`, `123`.

Đây là cách làm sai tầng: giới hạn RL thuộc về config Python, không phải file USD nhị phân.
USD không diff được bằng git, nên mỗi lần thử một giới hạn mới lại đẻ ra một file tên vô nghĩa.
Cách đúng: giữ một USD với giới hạn cơ khí thật, đổi giới hạn RL trong Python.

Hai file này vẫn mở được (tham chiếu đã trỏ lại `../../fulltrans/usd/FullForm/FullForm.usd`).

## `meshfixed_dupes/` — bản import trùng

`Fulltrans_meshfixed_{1,2,3}` là **3 lần import lại cùng một URDF**. File `.usda` gốc chỉ khác
chuỗi đường dẫn `/tmp/urdf_import_xxxxx` mà trình import ghi vào; payload chênh nhau 2–6 byte,
đúng bằng độ dài thêm của hậu tố `_1`/`_2`/`_3`. Bản đang giữ ở
`../fulltrans/usd/Fulltrans_meshfixed/`.

## `fulltrans_usd_dupe/` — bản sao khít từng byte

Từng nằm ở `FullForm/urdf/Fulltrans/`. Trùng md5 hoàn toàn với
`../fulltrans/usd/{Fulltrans10DOF.usd, configuration/}` — 10.5 MB nhân đôi.

## `axisnewbipedal/` — đời robot cũ

10 khớp, có `Rotate_left`/`rotate_right` thay cho Twist, tên khớp không nhất quán
(`Kneeright` thiếu hậu tố `_joint`). Mesh khác hẳn các họ khác. Không có USD, chưa từng
đưa vào Isaac.
