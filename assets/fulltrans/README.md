# fulltrans — họ robot chính (đang train)

Xuất từ SolidWorks bằng SW2URDF `1.6.0-4-g7f85cfe`, **2026-03-18 21:46** (xem `meta/export.log`).

## Ba biến thể URDF, dùng chung một bộ mesh

| File | Khớp | Khác biệt |
|---|---|---|
| `urdf/FullForm.urdf` | 8 | Bub, Hip, Knee, Foot × 2. **Không có Twist** |
| `urdf/Fulltrans.urdf` | 10 | thêm Twist × 2 |
| `urdf/Fulltrans_meshfixed.urdf` | 10 | **giống hệt `Fulltrans.urdf`, khác đúng 4 dòng**: dùng `Hipleft_fixed.STL` / `Hipright_fixed.STL` thay cho bản gốc |

`_fixed` là mesh hông đã bo lại. Việc này đổi **vỏ lồi va chạm** (convex hull) — thứ mà cả
MuJoCo lẫn PhysX dùng thay cho mesh thật khi tính va chạm — nên nó là khác biệt vật lý thật,
không phải chỉ thẩm mỹ.

## USD

| File | Khớp | Ghi chú |
|---|---|---|
| `usd/Fulltrans10DOF.usd` | 10 | **Bản đang train.** Tự chứa, dùng `usd/configuration/Fulltrans_*.usd` |
| `usd/FullForm.usd` | 8 | Vỏ bọc, tham chiếu `./FullForm/FullForm.usd` |
| `usd/Fulltrans_meshfixed/` | — | Bản USDA thô do trình import URDF của Isaac xuất. Code không nạp. |

## Giới hạn khớp cơ khí (đơn vị: độ)

| Asset | Bub | Hip | Twist | Knee | Foot |
|---|---|---|---|---|---|
| `Fulltrans10DOF.usd` | −14.32 / 179.81 | ±37.30 | ±179.91 | ±126.05 | −65.21 / 40.26 |
| `FullForm.usd` | ±10.03 | ±37.30 | — | ±126.05 | ±40.26 |

Đây là giới hạn **cơ khí**. Giới hạn RL hẹp hơn nhiều và nằm trong config Python, không ở đây.

## ⚠️ Hai quy ước thứ tự khớp — dễ nhầm

```
meta/joint_names_FullForm.yaml (phía ROS, 8 khớp):
  Bubleft, Hipleft, Kneeleft, Footleft, Bubright, Hipright, Kneeright, Footright
  → chân TRÁI hết, rồi chân PHẢI

Isaac (10 khớp, xem comment trong `isaac_rl/_archive/fulltrans/transformer_walk10dof_env.py`):
  Bub_L, Bub_R, Hip_L, Hip_R, Twist_L, Twist_R, Knee_L, Knee_R, Foot_L, Foot_R
  → theo CẶP trái-phải
```

Trộn nhầm hai thứ tự khi xuất quỹ đạo sang robot thật = sai khớp toàn bộ. Phần tử đầu của
file YAML là chuỗi rỗng `''` (quirk của SW2URDF) — parser phải bỏ.

Chưa có `joint_names` cho bản 10 khớp: thứ tự đó hiện chỉ tồn tại trong comment Python.
Chạy `isaac_rl/scripts/check_joint_order.py` để lấy thứ tự thật từ articulation.

## MuJoCo

`mjcf/Fulltrans_RL.xml` — 10 actuator, `meshdir="../meshes/"`. Trước 2026-09-09 file này trỏ
`meshdir` tới đường dẫn tuyệt đối của một máy khác nên không nạp được; đã sửa thành tương đối.
