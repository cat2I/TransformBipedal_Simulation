# assets/ — mô hình robot

Mỗi thư mục con là **một họ cơ khí** (một đời robot). Chia theo họ chứ không theo số DOF,
vì các biến thể trong cùng một họ dùng chung một bộ mesh.

| Thư mục | DOF | Dùng cho | Trạng thái |
|---|---|---|---|
| [`officialdesign/`](officialdesign/README.md) | 10 | Handoff 2026-09-30, `Transformer-Official-10DOF-Direct-v0` | robot mới |
| [`fulltrans/`](fulltrans/) | 8 & 10 | Task đang train (`Transformer-Walk10DOF*`) | **đang dùng** |
| [`newsimple/`](newsimple/) | 6 | Task 6DOF cũ (`model_349.pt`) | tham chiếu |
| [`trans3dof/`](trans3dof/) | 6 | Bản đơn giản hoá, hình khác hẳn | tham chiếu |
| [`_archive/`](_archive/) | — | Đời robot cũ + bản trùng lặp | không dùng |

## Bố cục chuẩn của một thư mục robot

```
<robot>/
├── urdf/     .urdf + .csv        mô tả xương, do SolidWorks xuất
├── meshes/   .STL                hình học từng bộ phận
├── usd/      .usd                bản Isaac Sim nạp
├── mjcf/     .xml                bản MuJoCo nạp (nếu có)
├── meta/     .yaml, export.log   siêu dữ liệu: thứ tự khớp, lai lịch
└── ros/      package.xml, launch bao bì ROS 1 (chưa dùng)
```

**Đừng tách `usd/*.usd` khỏi thư mục cạnh nó.** Các file USD tham chiếu nhau bằng đường dẫn
tương đối (`./FullForm/FullForm.usd`, `configuration/Fulltrans_sensor.usd`). Di chuyển lẻ một
file là vỡ tham chiếu, và vì USD là nhị phân nên git sẽ **không** báo cho bạn biết.

## Code lấy asset thế nào

Qua `asset_path("<tên file>")` trong
[`_asset_paths.py`](../transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/_asset_paths.py).
Hàm này nhận **tên file**, tự dò trong `assets/*/usd/`. Không hard-code đường dẫn ở config.

## Ranh giới: cái gì thuộc về đây, cái gì không

`assets/` giữ **sự thật vật lý** — robot làm được gì. Giới hạn khớp trong URDF/USD là *giới hạn
cơ khí*: phần cứng chịu được tới đâu.

*Giới hạn RL* (`servo_min`/`servo_max` — policy được phép ra lệnh tới đâu) và *initial pose*
**không** thuộc về đây, dù cả hai đều là "giới hạn khớp". Ví dụ Bub cơ khí quay được
−14.32°…+179.81°, nhưng khi train walk chỉ cho dùng ±15°.

> Cách kiểm tra: đổi một con số mà **không phải cầm tuốc-nơ-vít sửa robot thật**
> → nó không thuộc `assets/`.
