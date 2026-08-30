# Dựng tư thế & phát lại chuỗi đứng dậy bằng MuJoCo

Công cụ: [`mujoco_view.py`](mujoco_view.py) — mở model URDF trong MuJoCo, kéo slider để
dựng tư thế, lưu lại thành keyframe, rồi phát lại thành động tác.

Dùng MuJoCo cho khâu này vì nó khởi động ~1 giây (Isaac mất 40–60 giây) và có
sẵn slider cho từng khớp. Isaac vẫn là nơi train RL — MuJoCo chỉ là **giấy nháp**
để thiết kế tư thế.

---

## Chuẩn bị

```bash
conda activate isaaclab30
cd ~/Desktop/TransformBipedal_Simulation/vinh_test
```

⚠️ `mujoco` chỉ cài trong env `isaaclab30`, **không có** trong `base`. Quên
activate sẽ báo `ModuleNotFoundError: No module named 'mujoco'`.

---

## 1. Mở model

```bash
python mujoco_view.py Fulltrans_meshfixed
```

Luôn dùng **`Fulltrans_meshfixed`**, đừng dùng `Fulltrans`. Bản `Fulltrans` có
lỗi xuất mesh: file `Hipleft.STL` bị nướng dính cả link `Twist` vào trong (thừa
3434 tam giác, dài gấp đôi), làm hỏng va chạm ở vùng đầu gối.

Xem các model khác: `python mujoco_view.py --list`

---

## 2. Đọc hai bảng bên phải — quan trọng nhất

| Bảng | Là gì |
|---|---|
| **Control** | **LỆNH** bạn ra cho động cơ — *"tôi muốn khớp tới góc này"* |
| **Joint** | **GÓC THẬT** hiện tại của khớp — *"khớp đang ở đâu"* |

Động cơ sinh mô-men theo công thức:

```
mô-men = kp × (Control − Joint) − kv × vận_tốc
```

Lệch càng nhiều thì đẩy càng mạnh. Hết lệch thì hết đẩy.

### Cách dùng để kiểm tra tính khả thi

| Thấy gì | Nghĩa là |
|---|---|
| Control ≈ Joint | ✅ Bình thường, servo làm được |
| Lệch ~0.25° khi đứng yên | ✅ Bình thường — cần lệch chút mới sinh mô-men chống trọng lực |
| Control = 3.14 mà Joint = 0.71 | ❌ Có gì đó chặn — hết mô-men, hoặc chạm giới hạn khớp |

**Đây là bài kiểm tra khả thi miễn phí.** Kéo Control tới tư thế mong muốn, nhìn
Joint có đuổi theo không. Không đuổi kịp = robot thật cũng không làm nổi.

### Tầm quay từng khớp (ghi trong URDF)

```
Bub      -14.3° ... 179.8°     rộng 194°   ← rộng nhất
Hip      -37.3° ...  37.3°     rộng  75°   ← HẸP NHẤT, thanh trượt ngắn là bình thường
Twist   -179.9° ... 179.9°     rộng 360°
Knee    -126.1° ... 126.1°     rộng 252°
Foot     -65.2° ...  40.3°     rộng 106°
```

Hip chỉ ±37° là do thiết kế cơ khí, không phải lỗi.

---

## 3. Chế độ gương — kéo 5 slider thay vì 10

```bash
python mujoco_view.py Fulltrans_meshfixed --mirror
```

Chỉ kéo 5 slider có chữ **`left`**. 5 slider `right` tự chạy theo y hệt.

Không cần đổi dấu, vì URDF đã lo phần đối xứng trong trục quay:
`Bubleft axis = [-1,0,0]` còn `Bubright axis = [+1,0,0]`.

> Kiểm chứng: đặt cả hai = `+0.5 rad` → hai bàn chân ra `y = -0.2567` và
> `+0.2562`, đối xứng đẹp qua mặt phẳng dọc.

Slider `right` vẫn hiện trong bảng Control nhưng kéo vô ích — mỗi bước nó lại bị
ghi đè. Cứ coi như không có.

---

## 4. Ghi lại tư thế

```bash
python mujoco_view.py Fulltrans_meshfixed --mirror --poses getup_poses.json
```

Kéo slider ra tư thế ưng ý → **nhấn ENTER trong cửa sổ viewer** → ghi ngay.
Terminal xác nhận:

```
[luu] pose #1 -> getup_poses.json
```

Làm lại nhiều lần để được cả chuỗi. File tự nối thêm; mở lại lần sau vẫn giữ
pose cũ (`dang co 6 pose`).

### Định dạng file

```json
{
  "model": "Fulltrans_meshfixed",
  "poses": [
    { "name": "K1",
      "ctrl": { "Bubleft_joint": 2.16, "Bubright_joint": 2.16, ... } }
  ]
}
```

Ghi theo **TÊN khớp**, không theo số thứ tự. Bắt buộc phải vậy, vì MuJoCo và
Isaac đánh số khớp khác nhau:

```
MuJoCo:  Bubleft, Hipleft, Twistleft, Kneeleft, Footleft, Bubright, ...
Isaac:   Bubleft, Bubright, Hipleft, Hipright, Twistleft, ...
```

Chép theo chỉ số giữa hai bên là **sai hoàn toàn**.

Muốn đổi tên pose cho dễ nhớ thì sửa trường `"name"` trực tiếp trong file JSON.

---

## 5. Phát lại chuỗi

```bash
python mujoco_view.py Fulltrans_meshfixed --play getup_poses.json --seg 1.5
```

Nội suy tuyến tính giữa các pose, chạy lặp liên tục. Robot được đặt thẳng vào
tư thế đầu tiên nên không phải rơi từ trên xuống.

### `--seg` — số giây mỗi đoạn, phải cẩn thận

MuJoCo **không mô phỏng đường cong mô-men–tốc độ**. Servo thật quay càng nhanh
càng yếu, mô-men về 0 tại `3.67 rad/s`. Nên MuJoCo cho kết quả quá lạc quan.

Đo trên chuỗi đứng dậy 6 pose:

| `--seg` | Tổng | Tốc độ đỉnh | Mô-men thật còn lại | |
|---|---|---|---|---|
| 3.00s | 15.0s | 8% | 92% | ✅ |
| 1.50s | 7.5s | 17% | 83% | ✅ |
| 1.00s | 5.0s | 24% | 76% | ✅ |
| 0.70s | 3.5s | 35% | 65% | ✅ |
| 0.50s | 2.5s | 46% | 54% | ✅ |
| 0.35s | 1.8s | 66% | 34% | ⚠️ |
| 0.25s | 1.2s | 88% | 12% | ❌ |

**Dùng `--seg 1.0` đến `1.5`.** Nhanh hơn thì trong MuJoCo vẫn đẹp nhưng ngoài
đời servo hết lực, chắc chắn ngã.

---

## 6. Chuỗi đứng dậy hiện có

`getup_poses.json` — 6 keyframe, gõ lại từ ảnh chụp màn hình nên số đã bị làm
tròn (`0.868` chứ không phải giá trị đầy đủ). **Nên ghi đè lại bằng ENTER** để
có số chính xác.

| Keyframe | Bub | Twist | Knee | Foot | z thân | Bão hòa mô-men |
|---|---|---|---|---|---|---|
| K1 nằm bẹt, chân xoạc | 2.16 | 1.63 | -1.17 | -0.982 | 0.0335 | 9% |
| K2 gập gối lại | 2.16 | 1.63 | -1.94 | -0.282 | 0.0940 | 18% |
| K3 khép chân, ngồi xổm | 1.31 | 1.60 | -1.56 | 0.27 | 0.2503 | 14% |
| K4 nhổm lên | 0.868 | 1.60 | -1.56 | 0.7027 | 0.2846 | 8% |
| K5 duỗi chân | 0.428 | 1.51 | -0.77 | 0.325 | 0.3687 | 19% |
| K6 đứng thẳng | 0.0719 | 1.51 | -0.044 | -0.107 | 0.3983 | 27% |

`Hip = 0` ở cả 6 keyframe.

**Kết quả chạy cả chuỗi** (`--seg 1.5`):

```
z: 0.0335 m  ->  0.3959 m       thân nghiêng 0.1°
=> ĐỨNG LÊN ĐƯỢC ✅
```

Bão hòa mô-men cao nhất chỉ **27%** — còn dư rất nhiều.

### Một điểm cần nhớ cho bước RL

`Twist` giữ nguyên **~1.5 rad (≈86°)** suốt cả chuỗi, kể cả lúc đứng thẳng.
Nghĩa là tư thế đứng đích **không phải** tư thế `Twist = 0` mà
`TRANSFORMER_10DOF_CFG` đang để làm `init_state`.

Khi viết env RL, tư thế đích phải là tư thế này.

---

## 7. Chiều cao đứng chuẩn = 0.3968 m

Đo trong MuJoCo sau khi vá: `z = 0.3968 m`, biên độ rung `0.07 mm`.
Isaac đo độc lập ra `0.397 m`. **Hai engine lệch 0.2 mm** → con số này đáng tin.

⚠️ `height_reward()` trong `transformer_standup_env.py` đang đặt
`ideal_height = 0.43` (ghi chú là đo từ PyBullet). **Lệch 3.3 cm.** Để nguyên
thì robot sẽ học cách nhón gót để với tới con số đó. Sửa trước khi train.

---

## 8. Các cờ khác

| Cờ | Tác dụng |
|---|---|
| `--fixed` | Treo thân cố định trên không, không rơi — tiện xem hình dáng chân |
| `--seg N` | Số giây mỗi đoạn khi `--play` (mặc định 1.5) |
| `--export F.xml` | Xuất MJCF thay vì mở viewer |
| `--armature 0` | Tắt quán tính rotor (xem model thô, sẽ rung) |
| `--keep-overlaps` | Giữ va chạm giả (xem model thô, chân sẽ kẹt) |
| `--list` | Liệt kê model có sẵn |

Trong viewer: panel **Simulation** bên trái có nút **`Reset`** (phím `Backspace`)
để đưa robot về tư thế ban đầu khi bị kẹt.

---

## 9. Hai thứ file này đã tự vá cho model

Model đọc thẳng từ URDF chạy rất xấu. Chi tiết đầy đủ nằm trong docstring đầu
[`mujoco_view.py`](mujoco_view.py); tóm tắt:

**a) Khớp không có quán tính động cơ → rung 174 Hz.** URDF không có trường
`armature` nên MuJoCo để bằng 0, khớp nhẹ như không khí, bộ giải số đẩy quá đà
rồi đẩy ngược liên tục. Đó là rung của *thuật toán*, không phải của vật lý.
→ Đặt `armature = 0.08` (bằng giá trị Isaac dùng). Rung giảm từ `qvelRMS 13.2`
xuống `0.0025`.

**b) Bộ dò va chạm dùng VỎ LỒI, không dùng hình thật.** Giống bọc màng bọc thực
phẩm quanh vật rồi hút chân không — mọi hốc lõm bị lấp phẳng thành khối đặc.
Thân robot rỗng nhiều hốc nên vỏ lồi phình **5.6 lần** (1876 → 10539 cm³). Khối
phình vô hình đó chặn chân: kéo Bub tới 179.8° thì khớp kẹt ở 40.9°, dù hai mesh
thật lúc đó còn cách nhau **37.8 mm**.

→ Bỏ va chạm giữa các khối **gần nhau trên cùng một chân**. Quy tắc:

```
cùng một chân, gần nhau    ->  BỎ    (khớp đã ràng buộc rồi, va chạm vô nghĩa)
chân trái đụng chân phải   ->  GIỮ   (va chạm thật, cần để dựng dáng đúng)
```

Sau khi vá, kéo hết cỡ từng khớp:

```
Bub    179.8 -> 179.7°    (trước khi vá: kẹt ở 40.9°)
Twist  179.9 -> 179.9°
Foot    40.3 ->  40.1°
Hip     37.3 ->  35.6°    hết tầm khớp, URDF ghi vậy
Knee   126.1 -> 116.7°    CHẠM THẬT: bàn chân đá vào thân, mesh cách 3.2 mm
```

---

## 10. Bước tiếp theo

1. Ghi lại 6 keyframe bằng ENTER để có số chính xác (thay cho số làm tròn hiện tại)
2. Đem chuỗi sang Isaac kiểm tra lại — Isaac có đường cong mô-men–tốc độ thật, MuJoCo thì không
3. Nội suy chuỗi thành quỹ đạo tham chiếu, viết env RL cho policy bắt chước
4. Nhớ sửa `ideal_height` từ `0.43` xuống `~0.397`
5. Nhớ đổi tư thế reset của env từ tư thế đứng sang tư thế **nằm** (`_reset_idx` hiện đang reset về tư thế đã đứng sẵn)
