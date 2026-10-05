# OFFICIALdesign — handoff 2026-09-30

Task mới: **`Official-Walk-v0`**, train từ đầu.
Nguồn CAD gốc được giữ tại `sim_handoff /`. Các task và checkpoint robot cũ
không đổi. Checkpoint cũ không tương thích ngữ nghĩa dù có cùng shape 60/10.

## Cấu hình đã tích hợp

- `OFFICIALdesign/urdf/OFFICIALdesign.urdf`: sửa mass, CoM và đủ 6 phần tử
  tensor của **13 link** từ `meta/inertial_report.csv`. Tổng **4.643 kg**.
  CSV CoM mm → URDF m; inertia giữ kg·m², tại CoM, theo trục link.
  Không đổi dấu tensor lần nữa. `Bubright` đã có mass/inertia dương.
- `usd/OFFICIALdesign.usd`: asset chạy Isaac Lab 3/PhysX, floating base,
  10 revolute + 2 fixed IMU, self-collision bật, convex decomposition.
  Link được đưa về cùng cấp để Isaac Lab gắn sensor và thuộc tính đúng.
  Giữ nguyên tên, transform và quan hệ khớp của CAD.
- Converter dựng lại principal axes từ tensor CSV với quy ước ma trận Gf;
  kiểm tra tensor tái dựng thay vì chỉ kiểm tra eigenvalue. Bản importer
  `urdf-usd-converter 0.1.3` trên máy này chuyển eigenvectors sang Gf thiếu
  transpose, nên chỉ kiểm tra tổng mass sẽ không phát hiện sai inertia.
- `meta/calibration.json`: nguồn cấu hình actuator, dấu khớp, pose, phạm vi
  train và ma sát. `meta/build_report.json` lưu CoM, hình học đặt robot và
  hash nguồn. `meta/usd_report.json` gắn cấu hình với USD bằng SHA-256.

| Nhóm servo | Khớp | Effort/saturation | Velocity | Kp / Kd | Armature |
|---|---|---:|---:|---:|---:|
| STS3095 | Bub, Hip, Knee | 10.3 N·m | 4.7 rad/s | 60 / 1.5 | 0.01 kg·m² |
| STS3215 | rotate, Foot | 2.94 N·m | 4.7 rad/s | 40 / 1.0 | 0.01 kg·m² |

Torque là mức stall ước lượng trong handoff; velocity cũng là ước lượng.
Kp/Kd/armature và delay 10 ms là tham số khởi đầu của mô phỏng, chưa đo trên
servo thật. Ground/robot dùng static/dynamic friction 0.8/0.4, restitution 0.
Reset không ghi đè torque của servo nhỏ bằng giá trị servo lớn.

## Mapping và giao diện policy

Giữ quy ước tên CAD: **left nằm y âm**, tức phía phải của robot theo FLU.
`q_URDF_deg = sign × q_policy_deg`, zero offset = zero CAD.

| Index | Joint CAD | Sign | Pose policy (°) | Phạm vi policy tạm (°) |
|---:|---|---:|---:|---:|
| 0 | Bubleft_joint | +1 | 0 | −12 … 12 |
| 1 | Bubright_joint | −1 | 0 | −12 … 12 |
| 2 | Hipleft_joint | +1 | 20 | −15 … 45 |
| 3 | Hipright_joint | −1 | 20 | −15 … 45 |
| 4 | Nhat_rotate_left | +1 | 0 | −15 … 15 |
| 5 | rotate_right | +1 | 0 | −15 … 15 |
| 6 | Kneeleft_joint | +1 | −40 | −75 … 0 |
| 7 | Kneeright | −1 | −40 | −75 … 0 |
| 8 | Footleft_joint | +1 | 20 | −25 … 45 |
| 9 | Footright_joint | −1 | 20 | −25 … 45 |

Phạm vi trên là **training envelope**, không phải hard stop phần cứng đã đo.
Nó được đổi dấu và sắp xếp lower/upper trong cả URDF lẫn runtime. Pose chùng
gối mới có bàn chân phẳng; spawn z≈0.379647 m, cách sàn khoảng 5 mm theo STL.

Action 10D: clip [-1,1], tăng góc policy tối đa 2° mỗi bước, có clamp chống
tích lũy lệnh vượt giới hạn. Physics 200 Hz; policy **20 Hz** (decimation 10).
Observation 60D: 4 mẫu `[roll, pitch, gx, gy, gz]` của IMUleft rồi 4 mẫu lệnh
góc 10D đã normalize. Roll/pitch clip ±1 rad; gyro chia 2 rad/s và clip ±1.
Thứ tự lịch sử cũ → mới. Quaternion Isaac Lab 3 là xyzw, không dùng offset
IMU legacy. Nhiễu orientation 0.015 rad, gyro 0.01 rad/s chỉ cộng một lần.
Task học đi +x ở mục tiêu 0.15 m/s. Không thay đổi protocol Embedded trong repo khác.

## Chạy

Xem robot trực tiếp, sau khi bật môi trường Isaac trên máy này:

```bash
conda activate isaacsim
cd inspect                         # từ thư mục gốc dự án
python isaacsim_view.py --list
python isaacsim_view.py OFFICIALdesign
python isaacsim_view.py --fixed OFFICIALdesign
python isaacsim_inspect.py OFFICIALdesign
```

`view` mô phỏng robot với khớp thụ động; `--fixed` giữ cố định base.
`inspect` mở GUI để chỉnh khớp bằng Angular Drive → Target Position rồi nhấn
Play. Hai script dùng USD đã chuẩn bị; các drive phục vụ inspect chỉ nằm
trong stage xem tạm, không ghi đè asset train.

Từ thư mục `isaac_rl/`:

```bash
# Train/test ngắn
./run.sh scripts/validate_officialdesign.py --headless
./run.sh scripts/rsl_rl/train.py --task Official-Walk-v0 --num_envs 256 --headless --max_iterations 2

# Train thật; log riêng tại logs/officialdesign/
./run.sh scripts/rsl_rl/train.py --task Official-Walk-v0 --num_envs 256 --headless --max_iterations 3000

# Xem checkpoint mới (thay <run> và <iteration>)
./run.sh scripts/rsl_rl/play.py --task Official-Walk-v0 --num_envs 1 --checkpoint "$PWD/logs/officialdesign/<run>/model_<iteration>.pt"
```

Không cần convert lại để chạy asset đã bàn giao. Khi cập nhật CAD/calibration:

```bash
./run.sh scripts/prepare_officialdesign.py
./run.sh scripts/convert_officialdesign.py --headless --force
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./run.sh -m pytest scripts/test_officialdesign_asset.py -q
./run.sh scripts/validate_officialdesign.py --headless
```

Tắt pytest plugin autoload để tránh plugin ROS ngoài dự án đòi thư viện `lark`.
`--force` bảo đảm mesh thay đổi cũng được import lại. Các hash được kiểm tra khi
load task để tránh train với calibration/URDF khác USD.

## Các mục handoff chưa xác nhận

Nghiệm thu trên máy này: **4/4 offline tests**, GPU validation **8 env** pass
(giữ pose 5 giây không reset, contact, torque, partial reset và thả rơi),
PPO **256 env × 2 iterations**, tổng **12,288 steps**, exit code **0**.
Xem [`meta/validation_report.json`](meta/validation_report.json) và
[`meta/train_smoke_report.json`](meta/train_smoke_report.json).
Không còn lỗi tìm link/sensor; vẫn có các warning khởi tạo Isaac Lab được ghi
trong báo cáo. Đây là checkpoint smoke test, không phải policy đã học đi.

TODO: đo giới hạn góc, servo zero/offset, kiểm tra torque/speed theo điện áp
thực; kiểm tra CoM Hip hai bên lệch ~8 mm và Foot lệch ngang cùng dấu; chạy
lại script tensor mới trong SolidWorks. Dữ liệu báo cáo hiện có được dùng
nguyên trạng, không tự làm đối xứng. Smoke test chỉ nghiệm thu tích hợp chạy
được; không xác nhận policy đã biết đi hoặc đủ điều kiện triển khai phần cứng.

Tham chiếu API: [Isaac Lab URDF converter](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/api/lab/isaaclab.sim.converters.html),
[USD MassAPI](https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html).
