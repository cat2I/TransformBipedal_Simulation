# PLAN.md — Áp config đã đi bộ được (bản Isaac cũ) vào env chạy IsaacLab 3.0

## Công việc mới — OFFICIALdesign handoff 2026-09-30 (2026-10-01)

User yêu cầu tích hợp robot mới để train. Phạm vi đợt này độc lập với kế hoạch
NewSimple bên dưới: thêm `assets/officialdesign/`, script chuẩn bị/kiểm thử,
task `Transformer-Official-10DOF-Direct-v0`, đăng ký task và tài liệu chạy.
Giữ các thay đổi có sẵn của user. Không sửa dữ liệu handoff gốc.

- [x] Chuẩn bị URDF 13 link từ handoff: mass, CoM mm→m, tensor tại CoM;
      kiểm tra tensor vật lý và tổng 4.643 kg, mesh và cây 10 DOF.
- [x] Lưu mapping theo tên joint thật, dấu hai chân, giới hạn train tạm có
      nguồn rõ ràng; không gọi chúng là giới hạn cơ khí đã đo.
- [x] Import USD floating base, giữ IMU, collision convex, kiểm tra mass USD.
- [x] Task mới: actuator heavy/light riêng, reset sạch, lệnh theo tên khớp,
      IMU không nhân đôi, pose/chiều cao theo hình học robot mới, PPO riêng.
- [x] Chạy kiểm tra hướng khớp, reset, số hữu hạn, giới hạn torque/position,
      đứng giữ pose/thả rơi và train PPO ngắn trên GPU.
- [x] Ghi lệnh train/play và các mục CAD/phần cứng còn chưa xác nhận.

Giới hạn nghiệm thu: chạy được train không đồng nghĩa policy đã biết đi hay
thông số đã đủ để triển khai robot thật. Hip/Foot CoM và script SolidWorks
vẫn chờ kiểm tra phía CAD theo handoff.

Kết quả: offline 4/4; GPU validation 8 env pass (giữ pose 5 s, contact,
reset, torque, thả rơi); PPO 256 env × 2 iterations = 12,288 steps, exit 0.
Báo cáo: `assets/officialdesign/meta/validation_report.json` và
`assets/officialdesign/meta/train_smoke_report.json`.

---

Người lập: Claude (Tech Lead). Người thực thi: agy (Gemini). Ngày: 2026-09-10.

## Nguồn sự thật

Folder bản gốc của bạn user, train trên Isaac cũ, **robot đi bộ được**:
`/home/cat21/Documents/projects/Transformer/IsaacSim_TransformerBipedal/` (gọi tắt **REF**).
File quan trọng: `REF/transformer_nam/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/transformer_nam_env.py`
(gọi tắt **REF_ENV**). Git log của REF ghi rõ từng lần tune (2026-07-29 → 07-30).

Claude đã diff toàn bộ REF với repo này. Kết quả:
- `NewSimple.usd` + 4 layer `configuration/`: **md5 giống hệt** → asset không phải nguyên nhân.
- `transformer_config.py` (robot, actuator, pose): **giống** giá trị.
- `agents/rsl_rl_ppo_cfg.py`: **giống** giá trị, chỉ khác API rsl_rl 5.
- `transformer_nam_env.py`: **khác 8 chỗ** — toàn bộ việc nằm trong file này.

## Phạm vi

Chỉ sửa **1 file**:
`isaac_rl/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/transformer_nam_env.py` (gọi tắt **ENV**).

**Cấm sửa** mọi file khác. Trong ENV, **cấm đụng** những phần thuộc "port sang IsaacLab 3.0" (bảng B) —
đó là lý do REF_ENV không chạy được trực tiếp trên máy này.

---

## Bảng A — 8 giá trị phải đổi (REF_ENV là chuẩn)

| # | Vị trí trong ENV | Hiện tại | Đổi thành | Số dòng REF_ENV | Ghi chú |
|---|---|---|---|---|---|
| A1 | `EnvCfg.servo_max` | `(30, 30, -45, -45, 30, 30)` | `(35, 35, -40, -40, 35, 35)` | 131 | commit "tăng góc servo max ở knee" |
| A2 | `EnvCfg.servo_min` | `(20, 20, -55, -55, 20, 20)` | `(15, 15, -70, -70, 15, 15)` | 135 | commit "thay đổi giới hạn khớp" |
| A3 | `EnvCfg.weights["walk"]` | `[1, 1, 1, 0, 2, 1.2, 1]` | `[1, 1, 0.2, 0, 2, 6, 1]` | 62 | commit "tăng điểm velo" |
| A4 | `_setup_scene` → `RigidBodyMaterialCfg` | `static_friction=2.0, dynamic_friction=2.5` | `static_friction=0.8, dynamic_friction=0.4` | 209–210 | ma sát sàn |
| A5 | `_get_rewards` → `feet_height_reward(air_time, contact_pos, 0.03, 150)` | `0.03` | `0.04` | 313 | commit "tăng feetheight reward" |
| A6 | `joint_position_reward` → `max_diff` | `[5, 5, 5, 5, 5, 5]` | `[10, 10, 10, 10, 10, 10]` | 521 | commit "đổi độ lệch tối đa lên 10" |
| A7 | `_get_dones` → `height_termination` | `head_heights < 0.1` | `head_heights < 0.2` | 354 | commit "sửa điều kiện chết" |
| A8 | Khối IMU trong `EnvCfg` + `_get_observations` | có bias, drift, nhiễu 0.04/0.15, cờ legacy | xem Việc 2 | 223–227 | REF không có bias/drift |

`start_pos` giữ `(25, 25, -50, -50, 25, 25)` — REF dòng 142 giống.

## Bảng B — KHÔNG được đụng (port IsaacLab 3.0, REF_ENV không có vì chạy bản cũ)

| Thứ | Trong ENV | Trong REF_ENV | Lý do giữ bản ENV |
|---|---|---|---|
| `from ._lab3_compat import as_torch, imu_quat_w` | có | không | IsaacLab 3.0 trả `ProxyArray`, không phải Tensor |
| `as_torch(self.robot.data.…)` (mọi chỗ) | có | không | như trên |
| `imu_quat_w(self.robot, self.cfg.imu)` | có | `self.imu_data.quat_w` | như trên |
| `quaternion_to_euler`: `x, y, z, w = quat[...]` | xyzw | wxyz | IsaacLab 3.0 đổi quy ước quaternion |
| `from isaaclab.utils.configclass import configclass` | có | `from isaaclab.utils import configclass` | đường import 3.0 |
| `servo_max/min/start_pos` đọc từ `self.cfg.*` | có | hard-code | giữ, chỉ đổi giá trị mặc định (A1, A2) |

---

## Việc 1 — Áp A1..A7

Sửa đúng 7 con số/tuple theo Bảng A. Không sửa gì khác quanh đó.

- [x] AC1.1: `grep -n "servo_max\|servo_min" ENV` cho ra tuple đúng A1, A2.
- [x] AC1.2: `weights = {"walk": [1, 1, 0.2, 0, 2, 6, 1]}`. Thêm 1 dòng comment ngay trên:
      `# [orientation, height, joint_position, sigmoid_extra, feet_height, velocity, deviation] — theo REF 2026-07-30`
- [x] AC1.3: ma sát sàn `0.8 / 0.4`.
- [x] AC1.4: `feet_height_reward(..., 0.04, 150)`.
- [x] AC1.5: `max_diff = torch.tensor([10, 10, 10, 10, 10, 10], device=device)`.
- [x] AC1.6: `height_termination = head_heights < 0.2`.
- [x] AC1.7: Ngoài 7 dòng này, phần `_get_rewards`, `_get_dones` và các hàm sau `# REWARD FUNCTIONS` **không có diff nào khác**.

## Việc 2 — IMU về đúng REF (A8)

Trong `class TransformerWalkEnvCfg`:

| Trường | Hiện tại | Đổi thành |
|---|---|---|
| `domain_rand` | `True` | `False` |
| `imu_noise_std["orientation"]` | `0.04` | `0.015` |
| `imu_noise_std["angular_velocity"]` | `0.15` | `0.01` |
| `imu_drift_rate` | `0.0001` | `0.0` |
| `imu_fixed_bias` | `(0.0, -0.193, 0.0)` | `(0.0, 0.0, 0.0)` |
| `imu_legacy_double` | `True` | **xoá trường + xoá khối comment "TƯƠNG THÍCH NGƯỢC"** |
| `imu_bias_range` | giữ | giữ (chỉ có tác dụng khi `domain_rand=True`) |

Sửa comment cạnh `imu_noise_std`: bỏ "Tăng từ 0.015 → match real std", thay bằng
`# mặc định = REF (đã đi bộ được). Muốn tăng: env.imu_noise_std.orientation=0.04 trên dòng lệnh`.

Trong `_get_observations`, thay khối:
```python
orient_n = gaussian_noise(orient, GaussianNoiseCfg(...))
gyro_n = gaussian_noise(angular_vel_raw, GaussianNoiseCfg(...))
if self.cfg.imu_legacy_double:
    orient = orient + orient_n
    angular_vel = angular_vel_raw + gyro_n
else:
    orient = orient_n
    angular_vel = gyro_n
```
bằng đúng 2 dòng (đây chính là REF_ENV dòng 226–227, chỉ khác lấy std từ cfg):
```python
orient = gaussian_noise(orient, GaussianNoiseCfg(mean=0.0, std=self.cfg.imu_noise_std["orientation"]))
angular_vel = gaussian_noise(angular_vel_raw, GaussianNoiseCfg(mean=0.0, std=self.cfg.imu_noise_std["angular_velocity"]))
```
Rút khối comment "SỬA 2026-09-10" còn ≤ 3 dòng, ý: *`gaussian_noise()` đã trả về `data + nhiễu`; không cộng thêm lần nữa.*

Giữ nguyên: khối `if self._should_randomize_imu:` (bias random) và khối drift — chúng tắt khi giá trị mặc định = 0/False.

- [x] AC2.1: 5 giá trị đúng bảng.
- [x] AC2.2: `grep -n imu_legacy_double ENV` → 0 kết quả.
- [x] AC2.3: Trong `_get_observations` không còn `orient += ...` hay `orient + orient_n`.

## Việc 3 — Dọn comment sai đầu file

Khối comment trên `from .transformer_config import TRANSFORMER_CFG` (bắt đầu "transformer_config.py định nghĩa
TRANSFORMER_CFG HAI LẦN…") mô tả tình trạng cũ. Thay bằng:
```python
# TRANSFORMER_CFG = NewSimple.usd, 6 khớp, pose Hip 25° / Knee -50° / Foot 25°. Giống REF từng số.
# Khối FullForm111 (8 khớp) trong transformer_config.py đã comment — KHÔNG mở lại (lệch shape [1,6] != [1,8]).
```
- [x] AC3.1: Không còn chữ `transformer_config_3dof` trong ENV.

## Việc 4 — Kiểm chứng (agy chạy, dán output kèm diff)

```bash
cd isaac_rl
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk-Direct-v0 --num_envs 64 --headless --max_iterations 2
```
- [ ] AC4.1: Chạy hết 2 iteration không lỗi; log in `Observation dim: 44`, `Action dim: 6`.
- [ ] AC4.2: Trong `logs/rsl_rl/transformer_walk/<run mới>/params/env.yaml` có:
      `domain_rand: false`, `imu_drift_rate: 0.0`, `orientation: 0.015`, `angular_velocity: 0.01`,
      `servo_max: [35, 35, -40, -40, 35, 35]`, `servo_min: [15, 15, -70, -70, 15, 15]`,
      `weights.walk: [1, 1, 0.2, 0, 2, 6, 1]`.
- [ ] AC4.3: `git diff --stat` chỉ có `transformer_nam_env.py` (ngoài `PLAN.md`, `AGENTS.md`). Xoá run thử 2 iteration sau khi xác nhận; không commit log.

---

## Sau nghiệm thu — lệnh train thật (User chạy)

REF `agent.yaml`: `num_envs 512`, `max_iterations 350`. Run hôm qua (`2026-09-09_14-49-57`) chạy `num_envs 64`
= ít dữ liệu hơn **8 lần** mỗi iteration → không so sánh được. Bắt buộc 512:

```bash
cd isaac_rl
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk-Direct-v0 --num_envs 512 --headless --max_iterations 350
```
RTX 4050 6GB OOM → `--num_envs 256 --max_iterations 700` (giữ tổng số mẫu).

Play: `./run.sh scripts/rsl_rl/play.py --task Transformer-Walk-Direct-v0 --num_envs 1 --checkpoint "$PWD/logs/rsl_rl/transformer_walk/<run>/model_349.pt"`

Nếu sau 350 iter × 512 env vẫn không đi: config đã khớp REF 100% ở tầng Python → phần còn lại là backend
vật lý IsaacLab 2.x → 3.0. Lúc đó mới tune (ma sát sàn, `solver_position_iteration_count`, stiffness) — Claude lập plan mới.
