# Hướng dẫn Isaac Sim + IsaacLab — Transformer bipedal

Gộp ngày 2026-10-02 từ hai file cũ:
- `CAI-DAT-ISAACLAB.md`: về **máy**: phiên bản, tối ưu CPU/RAM/VRAM, số đo `num_envs`.
- `guidance.md`: về **quy trình**: sửa thưởng → train → play, bảng tra lỗi.

Số đo và kiểm chứng trong file này đo ngày 2026-08-03 trên env 10DOF cũ, trừ khi ghi khác. Thuật toán, hyperparameter và kế hoạch cho robot mới nằm ở `docs/ALGO.md`.

---

## Mục lục

0. [Ba mươi giây để chạy](#0-ba-mươi-giây-để-chạy)
1. [Máy đang có gì](#1-máy-đang-có-gì)
2. [Task đang dùng được](#2-task-đang-dùng-được)
3. [Vòng lặp chính: sửa thưởng → train → play](#3-vòng-lặp-chính-sửa-thưởng--train--play)
4. [Bảng lệnh tra nhanh](#4-bảng-lệnh-tra-nhanh)
5. [Checkpoint cũ](#5-checkpoint-cũ)
6. [Hiệu năng máy: CPU, RAM, VRAM, `num_envs`](#6-hiệu-năng-máy-cpu-ram-vram-num_envs)
7. [Mở cửa sổ Isaac Sim khi play](#7-mở-cửa-sổ-isaac-sim-khi-play)
8. [Khi gặp lỗi: tra ở đây](#8-khi-gặp-lỗi-tra-ở-đây)
9. [Những gì đã sửa và bài học](#9-những-gì-đã-sửa-và-bài-học)
10. [Dùng git để soát lại](#10-dùng-git-để-soát-lại)
11. [Tra cứu chéo](#11-tra-cứu-chéo)

---

## 0. Ba mươi giây để chạy

**Luôn đứng ở `isaac_rl/` trước.** `run.sh` nằm trong đó, không phải ở gốc repo. Đứng sai chỗ sẽ báo `bash: ./run.sh: No such file or directory`.

```bash
cd ~/Documents/projects/Transformer/Transform_bipedal_todai/isaac_rl
```

**Robot mới (OFFICIALdesign): train thử 3 vòng**

```bash
./run.sh scripts/rsl_rl/train.py --task Transformer-Official-10DOF-Direct-v0 \
    --headless --num_envs 4096 --max_iterations 3
```

> ⚠️ Env mới đặt `num_envs=256` trong code (`official_env.py:43`), nên **phải truyền `--num_envs 4096`**. Con số 4096 đo trên env cũ (mục 6). Robot mới có số link tương tự, nhưng nên đo lại một lần.

**Robot cũ (10DOF): play policy tốt nhất hiện có, có cửa sổ**

```bash
./run.sh scripts/rsl_rl/play.py --task Transformer-Walk10DOF-Direct-v0 --num_envs 1 \
    --checkpoint "$PWD/logs/rsl_rl/transformer_walk/2026-07-23_15-23-03/model_1499_rslrl5.pt"
```

Không cần thêm cờ: cửa sổ tự mở, tốc độ tự đúng thời gian thật. Lần đầu sau khi bật máy mất khoảng **2 phút**, những lần sau khoảng **23 giây** (mục 7).

**Không cần `conda activate`.** `run.sh` tự làm hết:
- kích hoạt env `isaacsim`;
- đặt `PYTHONPATH`;
- đặt biến môi trường cho GPU 6 GB;
- dùng `isaac-run` nếu đã cài.

`run_direct.sh` và `isaaclab.sh` chỉ chuyển tiếp sang `run.sh`.

> **Khác nhau cơ bản giữa train và play:**
> - Train luôn `--headless` (nhanh, tiết kiệm VRAM).
> - Play không truyền gì thì tự có cửa sổ.
> - Đừng thêm `--rendering_mode` vào play (mục 7.3).

---

## 1. Máy đang có gì

| Thành phần | Bản | Ghi chú |
|---|---|---|
| conda env | `isaacsim` | Python 3.12.13 |
| Isaac Sim | 6.0.1.0 | Cài bằng pip |
| IsaacLab | 3.0.0-beta2.patch1 | Cài editable từ `~/IsaacLab`. **Mặc định headless/GUI bị đảo so với 2.x** (mục 7) |
| rsl-rl-lib | 5.0.1 | API mới, actor và critic tách riêng |
| torch | 2.11.0+cu130 | CUDA nhận GPU |
| warp-lang | 1.13.0 | |
| numpy | 2.3.1 | |
| tensorboard / gymnasium / tqdm / hydra-core | có đủ | |
| transformer_nam | 0.1.0 | Cài editable |

Phần cứng: laptop RAM 16 GB, RTX 4050 6 GB VRAM, đồ họa lai (desktop chạy trên iGPU Intel).

**Không thiếu gói nào.** `pip check` báo 2 cảnh báo, cả hai vô hại và **cố ý không sửa**:

```
isaaclab-rl 0.5.5   yêu cầu packaging<24   (đang có 26.0)
isaacsim-kernel     yêu cầu coverage==7.4.4 (đang có 7.6.1)
```

Đây là ràng buộc phiên bản cũ ghi trong metadata (siêu dữ liệu của gói). `train.py` chỉ dùng `packaging` cho `version.parse`, chạy thật nhiều lần không lỗi. `coverage` chỉ dùng khi chạy test. Hạ cấp hai gói này rủi ro hơn để nguyên.

**Không có** thư mục `~/IsaacLab/isaac-sim`. Đó là cấu trúc của bản Isaac Sim đóng gói sẵn. Máy này cài bằng pip nên không có `python.sh` hay `setup_python_env.sh`. Script nào trỏ vào đó là script cũ.

---

## 2. Task đang dùng được

| Task | obs | act | File env | Ghi chú |
|---|---:|---:|---|---|
| `Transformer-Official-10DOF-Direct-v0` | 60 | 10 | `official_env.py` | **Robot mới.** Log ở `logs/rsl_rl/officialdesign_walk/` |
| `Transformer-Walk10DOF-Direct-v0` | 60 | 10 | `transformer_walk10dof_env.py` | Robot cũ, đủ 10 khớp |
| `Transformer-Walk10DOF6-Direct-v0` | 44 | 6 | `transformer_walk10dof6_env.py` | Robot cũ, RL chỉ điều khiển 6 khớp |

Các file env nằm ở `isaac_rl/source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/`. Đăng ký task ở `__init__.py` cùng thư mục. Các task khác đang bị comment out.

Bảng đối chiếu 81 run cũ với task tương ứng: `isaac_rl/logs/README-runs.md` (tóm tắt ở mục 5).

**Trạng thái training của robot cũ: chưa biết đi.**

| Run | Task | iter | ep_len | % của 200 |
|---|---|---:|---:|---:|
| `2026-07-23_15-23-03` | 10DOF | 1500 | 83.1 | **41.5%** |
| `2026-07-23_15-12-33` | 10DOF6 | 350 | 32.9 | 16.5% |

Run tốt nhất ngã ở giây thứ 4.2 trên 10. Trên asset thật, 10DOF đang nhỉnh hơn 10DOF6.

---

## 3. Vòng lặp chính: sửa thưởng → train → play

### Bước 1. Sửa hàm thưởng

Mọi thứ nằm trong **một file env** ứng với task bạn dùng.

#### Env mới: `official_env.py`

| Muốn sửa gì | Ở đâu |
|---|---|
| Các thành phần thưởng và hệ số | `_get_rewards()`, dòng **161**; công thức tổng ở dòng 175 |
| Điều kiện kết thúc sớm | `_get_dones()`, dòng **187**: thân < `min_base_height` 0.20 m, hoặc nghiêng > `max_tilt` 0.9 rad |
| Nội dung obs | `_get_observations()`, dòng 127 |
| Vận tốc mục tiêu, độ dài episode, `num_envs` | Cfg, dòng 59, 32, 43 |

Ở env mới, **hệ số là hệ số thật**: mỗi số nhân trực tiếp với thành phần của nó. Tăng một số thì chỉ thành phần đó mạnh lên. Bảng đầy đủ các thành phần xem ở `ALGO.md` mục 2.2.

#### Env cũ: `transformer_walk10dof_env.py` / `transformer_walk10dof6_env.py`

| Muốn sửa gì | Dòng (10DOF) | Dòng (6DOF) |
|---|---|---|
| Tỉ trọng 7 thành phần thưởng | **75–77** `weights = {...}` | 67–69 |
| Cách cộng thành tổng | **345** `_get_rewards()` | 344 |
| Điều kiện ngã / kết thúc sớm | **386** `_get_dones()` | 385 |
| Nội dung obs | 262 `_get_observations()` | 256 |
| Độ dài episode, số env | 44, 106 | 48, 93 |
| Công thức từng thành phần | **477–632** (các hàm `@torch.jit.script`) | tương tự |

```python
weights = {
    #        orient  height  position  sig_extra  feet_h  velocity  deviation
    "walk": [   1,      1,       1,         0,       2,      1.2,       1    ],
}
```

| # | Tên | Hàm tính | Ý nghĩa |
|---|---|---|---|
| 0 | orientation | `orientation_reward` (514) | Thân thẳng, phạt nghiêng |
| 1 | height | `height_reward` (558) | Giữ hông ở **0.43 m** |
| 2 | position | `joint_position_reward` (575) | Khớp gần tư thế gốc |
| 3 | sig_extra | `sigmoid_extra` (609) | **Đang tắt** (trọng số 0) |
| 4 | feet_h | `feet_height_reward` (619) | Nhấc chân đủ cao |
| 5 | velocity | `velocity_reward` (588) | Đi đúng hướng |
| 6 | deviation | `deviation_reward` (533) | Không lệch khỏi đường thẳng |

Env cũ kết thúc khi hông < 0.1 m hoặc |roll|, |pitch| > 0.95 rad.

#### ⚠ Ba cái bẫy

**Bẫy 1 (chỉ env cũ): trọng số là TỈ LỆ, không phải hệ số.** Dòng 361:
```python
w = self.weights / torch.sum(self.weights, dim=1, keepdim=True)
```
- Bảy số được chia cho tổng của chúng. Tổng hiện tại là 7.2, nên `feet_h` chiếm 2/7.2 = 27.8%.
- Tăng `feet_h` từ 2 lên 4 **không** làm nó mạnh gấp đôi. Nó làm **mọi thành phần khác yếu đi**.
- Muốn tăng riêng một thành phần thì phải giảm các số còn lại, hoặc sửa thẳng công thức.

**Bẫy 2 (mọi env): `robot.data.*` phải bọc `as_torch()`.**
- IsaacLab 3.0 trả về `ProxyArray` (một kiểu mảng riêng của IsaacLab), không phải `torch.Tensor`.
- `ProxyArray` **không đi qua được `@torch.jit.script`** (cơ chế biên dịch hàm PyTorch cho nhanh, kiểm tra kiểu rất chặt).

```python
# ĐÚNG
robot_root_pos = as_torch(self.robot.data.root_pos_w)
# SAI: nổ khi truyền vào hàm @torch.jit.script
robot_root_pos = self.robot.data.root_pos_w
```

`as_torch` nhập từ `._lab3_compat`, đã có sẵn ở đầu mỗi file env. Jit cũng **không nhận** `Optional`, `dict` hay kiểu động. Tham số nào không phải Tensor thì phải chú thích rõ kiểu (`action: str`, `target_h: float`).

**Bẫy 3 (mọi env): đổi số chiều obs hoặc action là mất hết checkpoint cũ.**
- Mạng nơ-ron sẽ sai kích thước. Đây không phải lỗi: chấp nhận train lại từ đầu, hoặc giữ nguyên số chiều.
- Sửa **công thức thưởng** thì không sao. Checkpoint vẫn nạp được, chỉ là policy cũ được chấm theo thước đo mới.

### Bước 2. Kiểm tra cú pháp (2 giây, tiết kiệm 40 giây)

Isaac Sim mất khoảng 40 giây để khởi động. Đừng để nó khởi động xong mới báo lỗi thụt lề.

```bash
python -m py_compile source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/*.py && echo OK
cd .. && git diff --stat && cd isaac_rl     # soát lại thay đổi
```

### Bước 3. Chạy thử 3 vòng trước khi train thật

```bash
./run.sh scripts/rsl_rl/train.py --task <TASK> --headless --max_iterations 3
```

| Dấu hiệu | Nghĩa là |
|---|---|
| `Mean reward` ra số bình thường | Thưởng tính được |
| `Mean reward: nan` | Công thức chia cho 0 hoặc `log` số âm. Sửa ngay |
| `Mean episode length` > 10 | Robot không ngã tức thì |
| Không có `Traceback` | Không lỗi kiểu dữ liệu |
| Không có dòng chứa `overflow` | Buffer PhysX đủ dùng |

Nếu ra `nan` mà không thấy lỗi, tăng dần `--max_iterations` để tìm vòng nào bắt đầu hỏng.

### Bước 4. Train thật

**Train từ đầu** (khi đã đổi thưởng đáng kể):

```bash
./run.sh scripts/rsl_rl/train.py --task <TASK> --headless --max_iterations 1500
```

**Train tiếp từ checkpoint** (khi chỉ tinh chỉnh nhẹ):

```bash
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 \
    --headless --max_iterations 1500 \
    --resume --load_run 2026-07-23_15-23-03 --checkpoint model_1499_rslrl5.pt
```

- `num_envs`: env cũ mặc định đã là 4096. Env mới phải truyền `--num_envs 4096`.
- Muốn nhanh thêm khoảng 10%: đóng hết ứng dụng rồi dùng `--num_envs 8192`. Trên 8192 không nhanh thêm (mục 6).
- **Thời gian:** 41 304 steps/s ở 4096 env, khoảng **2.3 giây mỗi vòng**, nên 1500 vòng ≈ **1 giờ**.
- Checkpoint tự lưu mỗi 50 vòng vào `logs/rsl_rl/<experiment>/<ngày-giờ>/model_<N>.pt`.

> **Khi resume:** vòng đầu `mean_episode_length` **tụt mạnh** rồi mới leo lại (ví dụ 83 → 24 → 70 sau 10 vòng).
> Nguyên nhân: `init_at_random_ep_len=True` trong `train.py`, cộng với Adam (thuật toán tối ưu) khởi động lại từ trạng thái trống. Cả hai tự hết sau khoảng 10 vòng.
> **Đừng tưởng là hỏng.** Sau 20 vòng vẫn không leo lại thì mới đáng nghi.

### Bước 5. Theo dõi bằng TensorBoard

Mở **terminal thứ hai**:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacsim
cd ~/Documents/projects/Transformer/Transform_bipedal_todai/isaac_rl
tensorboard --logdir=logs/rsl_rl/<experiment> --port=6006
```

`<experiment>` là `officialdesign_walk` (robot mới) hoặc `transformer_walk` (robot cũ). Mở trình duyệt vào `http://localhost:6006`.

**Đường quan trọng nhất: `Train/mean_episode_length`.** Episode tối đa = 10 s ÷ (0.005 × 10) = **200 bước**.

| Giá trị | Nghĩa |
|---|---|
| ~18 | Policy ngẫu nhiên, chưa học được gì |
| 83 | Mức tốt nhất của robot cũ: ngã ở giây 4.2 |
| 200 | Sống trọn 10 giây: mục tiêu |

`Train/mean_reward` chỉ so được giữa hai lần chạy **cùng hàm thưởng**. `mean_episode_length` thì so được mọi lúc, vì nó đo hành vi thật chứ không đo thước đo. Các chỉ số khác xem `ALGO.md` mục 1.7.

### Bước 6. Play thử

```bash
./run.sh scripts/rsl_rl/play.py --task <TASK> --num_envs 1 \
    --checkpoint "$PWD/logs/rsl_rl/<experiment>/<run>/model_<N>.pt"
```

- Tự mở cửa sổ và chạy đúng thời gian thật.
- Muốn chạy ngầm, chỉ xem log: thêm `--headless`.
- Lần đầu mất khoảng 2 phút và **trông y như máy treo. Đừng tắt** (mục 7.1).

> ### ⚠ Khác biệt dễ vấp nhất
> | Script | `--checkpoint` nhận |
> |---|---|
> | `train.py --resume` | **tên file**: `model_1499_rslrl5.pt` |
> | `play.py` | **đường dẫn đầy đủ**: `$PWD/logs/.../model_1499_rslrl5.pt` |
>
> Đưa tên file trần cho `play.py` sẽ báo `FileNotFoundError`. Không phải lỗi project.

**Đọc log play (env cũ):**

```
[ 1348] roll=+0.172 pitch=-0.632 gz=-0.290 h=0.397m | TwL=+23.7° TwR=+5.1° | air L=0.45s R=0.00s contact L=n R=Y
```

| Trường | Nghĩa | Muốn thấy gì |
|---|---|---|
| `h=` | Độ cao hông | Quanh **0.43 m**; dưới 0.1 là ngã |
| `roll` / `pitch` | Độ nghiêng (rad) | Gần 0; vượt **0.95** là kết thúc |
| `TwL` / `TwR` | Góc khớp xoay hai chân | Dao động đều = đang bước |
| `air L/R` | Thời gian chân treo | **So le nhau** = bước thật, không phải nhảy cóc |
| `contact L/R` | Chạm đất | Luân phiên Y/n |

Hai chân `air` cùng tăng = robot đang nhảy. Cả hai cùng 0 = đang đứng yên.

> ⚠ **Log sinh trước 03-08-2026 có cột `TwL`/`TwR` đọc nhầm khớp.** Thứ tự khớp thật **xen kẽ** trái/phải:
> ```
> 0 Bubleft  1 Bubright  2 Hipleft  3 Hipright  4 Twistleft
> 5 Twistright 6 Kneeleft 7 Kneeright 8 Footleft 9 Footright
> ```
> `play.py` cũ viết cứng index 2 và 7, tức là Hipleft và Kneeright, không phải hai khớp xoay. **Đừng dùng log cũ để đánh giá chuyển động xoay.** Đã sửa: giờ tra theo tên và in `Twistleft=[4] Twistright=[5]` lúc khởi động.

Env mới in thứ tự khớp và chỉ số Isaac lúc khởi động (`official_env.py:104-105`). Hãy đối chiếu dòng đó mỗi khi đổi asset.

### Bước 7. Xuất policy cho robot thật

`play.py` **tự động** xuất khi chạy:

```
logs/rsl_rl/<experiment>/<run>/exported/policy.pt     ← TorchScript
logs/rsl_rl/<experiment>/<run>/exported/policy.onnx   ← ONNX (dùng trên Pi)
```

Hai file này là mạng đã đóng gói, chạy không cần Isaac Sim.

---

## 4. Bảng lệnh tra nhanh

Tất cả chạy từ `isaac_rl/`.

| Việc | Lệnh |
|---|---|
| Thử nhanh 3 vòng | `./run.sh scripts/rsl_rl/train.py --task <TASK> --headless --max_iterations 3` |
| Train robot mới | `./run.sh scripts/rsl_rl/train.py --task Transformer-Official-10DOF-Direct-v0 --headless --num_envs 4096 --max_iterations 3000` |
| Train robot cũ | `./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 --headless --max_iterations 1500` |
| Train tiếp | thêm `--resume --load_run <thư-mục> --checkpoint <tên-file>` |
| Play (có cửa sổ) | `./run.sh scripts/rsl_rl/play.py --task <TASK> --num_envs 1 --checkpoint "$PWD/logs/rsl_rl/<exp>/<run>/model_<N>.pt"` |
| Play không cửa sổ | thêm `--headless` |
| TensorBoard | `tensorboard --logdir=logs/rsl_rl/<exp> --port=6006` |
| Chuyển checkpoint cũ sang định dạng mới | `./run.sh scripts/convert_checkpoint_rslrl5.py logs/rsl_rl/transformer_walk/<run>` |
| Kiểm tra cú pháp | `python -m py_compile source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/*.py` |
| Xem CPU/RAM/swap | `isaac-perf status` |
| Xem run nào dùng task nào | `cat logs/README-runs.md` |
| Soát thay đổi | `cd .. && git diff` |

---

## 5. Checkpoint cũ

### 5.1 Phải chuyển đổi trước khi play

- Checkpoint train bằng rsl_rl bản cũ lưu chung một `model_state_dict`.
- rsl-rl-lib 5.0.1 đòi `actor_state_dict` và `critic_state_dict` tách rời. Nạp file cũ sẽ báo `KeyError: 'actor_state_dict'`.
- Kiến trúc mạng không đổi, chỉ khác tên khóa, nên chuyển đổi được:

```bash
./run.sh scripts/convert_checkpoint_rslrl5.py logs/rsl_rl/transformer_walk/<tên-run>
```

File gốc giữ nguyên, bản mới ghi ra `<tên>_rslrl5.pt` bên cạnh.

- **Đã chuyển hết 488 checkpoint của 81 run** (2026-08-03).
- 66 file `exported/policy.pt` không chuyển, vì đó là model JIT đã xuất, không phải checkpoint train.
- Optimizer trong file chuyển đổi là bản trống. Play không dùng optimizer nên không sao. Train tiếp thì Adam chỉ mất vài chục bước để dựng lại.

### 5.2 Đã kiểm chứng: checkpoint cũ còn nguyên giá trị

**Câu hỏi:** IsaacLab 3.0 đổi quy ước quaternion từ `(w,x,y,z)` sang `(x,y,z,w)` và đổi sensor IMU. Nếu lớp bù trong `_lab3_compat.py` sai thì cả 488 file là rác.

**Thí nghiệm đối chứng:** hai lần train 10 vòng, cùng `--num_envs 4096`, chỉ khác có `--resume` hay không.

| | ep_len đầu → cuối | reward đầu → cuối | action std |
|---|---|---|---|
| Train **từ đầu** | 18.5 → 18.0 *(đứng yên)* | 2.44 → 2.52 | 1.01 |
| **Resume** `model_1499` | 23.9 → **69.7** | 2.93 → **7.60** | 4.04 |

Bản resume đạt episode length gấp **3.9 lần**. Nếu policy hỏng thì kết quả phải trùng cột "từ đầu". **Kết luận: lớp bù đúng, dữ liệu cũ dùng được.** Kiểm tra thêm trên file: `std`, toàn bộ trọng số và `iter` của bản `_rslrl5.pt` trùng khít với file gốc.

### 5.3 Run nào play được với task nào

| obs / act | Task | Số run | Checkpoint |
|---|---|---:|---:|
| 44 / 6 | `Transformer-Walk10DOF6-Direct-v0` | 45 | 341 |
| 60 / 10 | `Transformer-Walk10DOF-Direct-v0` | 4 | 48 |
| 84 / 10 | *chưa có task khớp* | 4 | 12 |
| 12 / 6 | *chưa có task khớp* | 11 | 39 |
| 14 / 8 | *chưa có task khớp* | 12 | 34 |
| 16 / 8 | *chưa có task khớp* | 2 | 5 |
| 54 / 8 · 56 / 8 · 62 / 10 | *chưa có task khớp* | 3 | 9 |

**49/81 run play được ngay.** 32 run còn lại là thí nghiệm cũ (tháng 2–5) với số khớp khác. Muốn play thì bật lại env tương ứng trong `__init__.py`.

> ⚠ Checkpoint robot cũ **không dùng được** cho robot mới, dù cùng 60/10. Asset, thứ tự khớp, biên khớp và ý nghĩa obs đều khác.

Bốn run 10DOF, mới nhất trước:

| Run | Checkpoint | Ghi chú |
|---|---:|---|
| `2026-07-27_15-02-49` | 3 | Dừng ở iter 99 |
| `2026-07-23_15-23-03` | **31** | **Tốt nhất, tới `model_1499`** |
| `2026-07-23_14-43-46` | 11 | Tới iter 499 |
| `2026-07-23_14-33-32` | 3 | Tới iter 99 |

---

## 6. Hiệu năng máy: CPU, RAM, VRAM, `num_envs`

### 6.1 Tối ưu CPU (chạy lại một lần)

```bash
sudo bash ~/isaac-setup.sh
```

- Gõ mật khẩu đăng nhập. **Không hiện ký tự nào là bình thường.**
- Script idempotent (chạy lại bao nhiêu lần cũng cho cùng kết quả, an toàn).
- Nó nâng swapfile 8 → 16 GB, zram 8 → 12 GB, chỉnh sysctl, và cài 2 lệnh `isaac-perf` / `isaac-run`. Sau đó `run.sh` tự dùng `isaac-run`.

```bash
isaac-perf status     # xem CPU / RAM / swap / zram / trạng thái daemon
isaac-perf on         # bật hiệu năng cao thủ công
isaac-perf off        # trả về tiết kiệm điện
```

Kiểm tra bất cứ lúc nào: `powerprofilesctl get` ra `performance` là đúng. Nếu ra `balanced` thì chạy `powerprofilesctl set performance` (không cần sudo).

**Vì sao phải chạy lại (sửa 2026-08-03).** Bản `isaac-perf` đầu tiên làm máy tụt hiệu năng âm thầm vì 2 lỗi:
1. Ghi thẳng `/sys/firmware/acpi/platform_profile` trong khi `power-profiles-daemon` cũng quản lý file đó, nên hai bên ghi đè nhau.
2. Lưu trạng thái gốc ở `/run/isaac-perf.state`, mà `/run` bị xóa mỗi lần khởi động. Mất state thì `isaac-perf off` rơi về giá trị đoán sẵn (`balanced`), trong khi máy này đúng ra là `performance`.

Bản mới lưu state ở `/var/lib/isaac-perf.state` (sống qua reboot), bỏ giá trị đoán sẵn, và đi qua `powerprofilesctl`.

### 6.2 `num_envs`: số đo thật (env 10DOF cũ, 2026-08-03)

Đo bằng train thật 5 vòng, headless. `steps/s = num_envs × 24 / thời gian mỗi vòng`.

| num_envs | VRAM mặc định | VRAM đã cắt buffer | steps/s |
|---:|---:|---:|---:|
| 512 *(số cũ)* | 2437 MiB | — | 14 288 |
| 1024 | 2633 MiB | — | 24 824 |
| 2048 | 2953 MiB | — | 36 141 |
| 4096 | 3547 MiB | **1897 MiB** | 41 304 |
| 8192 | 4803 MiB | **3153 MiB** | 45 406 |
| 12288 | — | 4367 MiB | 45 511 *(không tăng nữa)* |

**Chọn 4096, không chọn 8192:**

| num_envs | VRAM | RAM thấp nhất khi chạy | steps/s |
|---:|---:|---:|---:|
| **4096** | 1897 MiB | **6604 Mi** | 41 304 |
| 8192 | 3153 MiB | 4341 Mi | 45 406 (+10%) |

8192 nhanh hơn 10% nhưng ăn thêm 2263 Mi RAM dự phòng. Mở Firefox (khoảng 4 GB) là 8192 hết chỗ, còn 4096 vẫn dư khoảng 2.6 Gi. Task 6DOF cho số gần trùng khít.

### 6.3 Buffer PhysX đã cắt

Đặt trong `SimulationCfg(physics=PhysxCfg(...))` của các file env, **kể cả `official_env.py`**. Cắt buffer tiết kiệm **1650 MiB ở mọi mức `num_envs`**, vì đó là cấp phát cố định.

| Tham số | Mặc định | Đặt lại | Lý do |
|---|---|---|---|
| `gpu_max_soft_body_contacts` | 2²⁰ | 2¹⁰ | Scene không có soft body |
| `gpu_max_particle_contacts` | 2²⁰ | 2¹⁰ | Scene không có particle |
| `gpu_max_rigid_contact_count` | 2²³ | 2²⁰ | Chỉ 2 bàn chân chạm đất mỗi env |
| `gpu_collision_stack_size` | 2²⁶ | 2²⁴ | Như trên |
| `gpu_temp_buffer_capacity` | 2²⁴ | 2²³ | Như trên |
| `gpu_found_lost_aggregate_pairs_capacity` | 2²⁵ | 2²² | Chỉ có robot + mặt đất |

**Không đụng** `gpu_found_lost_pairs_capacity`, `gpu_total_aggregate_pairs_capacity`, `gpu_max_rigid_patch_count`, `gpu_heap_capacity`, vì chúng co giãn theo `num_envs`.

Nếu thêm vật thể vào scene mà PhysX báo overflow: nó **tự in con số nó cần**. Nâng đúng tham số đó lên nấc 2ⁿ kế tiếp, đừng nâng cả bảng.

### 6.4 Chống lag, tràn RAM

Theo thứ tự hiệu quả:
1. **Luôn train với `--headless`.**
2. **Đóng Firefox nếu muốn dùng `--num_envs 8192`.** Ở 4096 thì không cần.
3. Khi play, để `--num_envs 1`.

Hai điều **không** giúp gì (đã đo):
- **Swap không nâng được `num_envs`.** Trần nằm ở bộ nhớ PhysX cấp phát sẵn. Nếu training phải swap thật thì tốc độ sập chứ không chạy thêm env. Hiện 28 Gi swap, dùng 0 B.
- **Đóng ứng dụng không giải phóng VRAM.** Desktop chạy trên iGPU Intel, RTX 4050 gần như trống sẵn. Đóng app giúp RAM, không giúp VRAM.

Lỗi "tràn RAM" trước đây gần như chắc chắn do buffer PhysX mặc định quá lớn cộng `num_envs` cao, không phải do thiếu swap.

---

## 7. Mở cửa sổ Isaac Sim khi play

### 7.1 Lần đầu rất lâu: đừng tắt

Số đo thật (2026-08-03), từ lúc gõ lệnh tới lúc robot bắt đầu bước:

| Lần chạy | Thời gian |
|---|---|
| **Đầu tiên** (cache nguội, sau khi bật máy) | **124 giây** |
| **Những lần sau** (cache đã ấm) | **23 giây** |

- Trong hai phút đó, Kit (nền tảng chạy Isaac Sim) nạp 243 extension rồi khởi tạo renderer RTX. Nó **không in gì** ra terminal, nên nhìn từ ngoài không phân biệt được "đang chạy" với "treo".
- Biết là xong khi terminal in dòng đầu tiên kiểu `[    0] roll=+0.156 pitch=-0.226 ...`.
- VRAM khi play có cửa sổ, `--num_envs 1`: khoảng 2600 MiB / 6144, GPU khoảng 37%.

### 7.2 IsaacLab 3.0 đảo ngược mặc định cửa sổ

| | IsaacLab 2.x | IsaacLab 3.0 (đang cài) |
|---|---|---|
| Mặc định | **có cửa sổ** | **headless** |
| Tắt cửa sổ | `--headless` | mặc định, hoặc `--viz none` |
| Bật cửa sổ | mặc định | **`--viz kit`** |

- `--headless` giờ chỉ là cờ deprecated (vẫn nhận, nhưng hết tác dụng). **Bỏ nó đi không còn làm hiện cửa sổ.**
- Thủ phạm: `~/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py`, hàm `_resolve_headless_settings`. Không chọn visualizer nào thì nó ép `headless = True`.
- `play.py` đã sửa để tự gán `--viz kit` và bật `real_time` khi bạn không nói gì. Truyền `--viz <tên>` thì `play.py` không can thiệp nữa, và `--real-time` cũng không tự bật.

### 7.3 Viewport ĐEN: đừng dùng `--rendering_mode`

**Triệu chứng:** cửa sổ hiện bình thường, cây Stage đủ `World`/`physicsScene`, log robot chạy đều, nhưng khung 3D **đen thui**, không thấy cả lưới sàn.

**Nguyên nhân:** chạy kèm `--rendering_mode performance`. Preset đó (`~/IsaacLab/apps/rendering_modes/performance.kit`) tắt hàng loạt tính năng RTX: ambient occlusion, reflections, indirect diffuse, hạ `maxBounces` xuống 2. Trên máy này kết quả là **không dựng được gì**.

**Cách sửa:** đừng truyền cờ đó. Mặc định của Kit hiện đủ và vẫn mượt. Đã kiểm chứng bằng chụp màn hình đối chứng: cùng checkpoint, có cờ thì đen, bỏ cờ thì robot hiện rõ.

### 7.4 Hộp thoại "Isaac Lab is not responding" hiện lặp lại

**Không phải Isaac Sim hỏng.** GNOME ping cửa sổ theo chu kỳ, quá `check-alive-timeout` (mặc định **5000 ms**) mà không trả lời thì báo treo. Kit đứng lâu hơn thế liên tục suốt 2 phút khởi động.

```bash
gsettings set org.gnome.mutter check-alive-timeout 0      # 0 = tắt kiểm tra treo
gsettings reset org.gnome.mutter check-alive-timeout      # hoàn tác
```

Đây là thiết lập **desktop**, áp cho mọi ứng dụng: app treo thật cũng sẽ không được báo nữa. Muốn giữ cảnh báo cho app khác thì đặt giá trị lớn, ví dụ `60000` (60 giây).

---

## 8. Khi gặp lỗi: tra ở đây

Các lỗi này đã gặp thật và đã sửa. Gặp lại thì tra bảng trước.

| Thông báo lỗi | Nguyên nhân gốc | Xử lý |
|---|---|---|
| `bash: ./run.sh: No such file or directory` | Đang đứng ở gốc repo | `cd isaac_rl` rồi chạy lại |
| Log bình thường **nhưng không mở cửa sổ** | IsaacLab 3.0 đảo mặc định | `play.py` đã tự thêm `--viz kit`. Script khác thì tự truyền cờ đó (7.2) |
| *"Isaac Lab is not responding"* lặp lại | GNOME timeout 5 giây | `gsettings set org.gnome.mutter check-alive-timeout 0` (7.4) |
| Cửa sổ mở, log chạy, **khung 3D đen** | `--rendering_mode performance` | Bỏ cờ đó (7.3) |
| `ImportError: cannot import name 'as_torch'` | Ai đó xóa `as_torch` khỏi `_lab3_compat.py` | Khôi phục hàm, **đừng** xóa dòng import |
| Lỗi kiểu dữ liệu lạ trong hàm `@torch.jit.script` | `ProxyArray` lọt vào jit | Bọc `as_torch(...)` (Bẫy 2, mục 3) |
| `KeyError: 'actor_state_dict'` | Checkpoint rsl_rl < 4.0 | `convert_checkpoint_rslrl5.py` (5.1) |
| `'PPO' object has no attribute 'actor_critic'` | Code viết cho rsl_rl < 4.0 | Dùng `runner.alg.get_policy()` |
| `IndexError: index 60 is out of bounds` | Code viết cứng cho task 62 chiều | Đọc số chiều từ `cfg` |
| `FileNotFoundError: ... model_99.pt` | `play.py` cần **đường dẫn đầy đủ** | Thêm `$PWD/logs/...` |
| `ModuleNotFoundError: No module named 'omni.ext'` | `omni.ext` chỉ nạp được **sau** khi SimulationApp khởi động | Đừng import ở mức module |
| Log có chữ `overflow` / `capacity` | Buffer PhysX thấp quá | Nâng đúng tham số PhysX in ra lên nấc 2ⁿ kế tiếp (6.3) |
| `size mismatch` khi nạp checkpoint | Đổi số chiều obs/action, hoặc nạp checkpoint robot cũ vào robot mới | Train lại, hoặc dùng đúng task (Bẫy 3, 5.3) |

**Robot ngã ngay sau khi sửa thưởng** (`mean_episode_length` khoảng 15 và không lên): kiểm tra `_get_dones()`. Sửa thưởng làm robot khom thấp dưới ngưỡng chiều cao thì bị kết thúc ngay dù chưa thật sự ngã.

---

## 9. Những gì đã sửa và bài học

Sao lưu toàn bộ code trước khi sửa: `isaac_rl/.backup_truoc_khi_sua_20260803_010207/`. Nhóm theo **nguyên nhân gốc** để rút ra bài học dùng lại được.

### Nhóm A. IsaacLab 2.x lên 3.0 đổi API

| Đổi cái gì | Hậu quả | Sửa thế nào |
|---|---|---|
| `robot.data.*` trả `ProxyArray` | Không lọt qua `@torch.jit.script` | Bọc `as_torch()` ở 56 chỗ trong 7 file |
| Quaternion đổi `(w,x,y,z)` → `(x,y,z,w)` | Góc nghiêng tính sai hoàn toàn | Sửa `quaternion_to_euler`, thêm cờ bù trong `_lab3_compat.py` |
| Sensor IMU mất `quat_w` | Không đọc được hướng thân | Hàm bù `imu_quat_w()` |
| `PhysxCfg` chuyển sang `isaaclab_physx` | Import cũ gãy | `from isaaclab_physx.physics import PhysxCfg` |
| Mặc định cửa sổ bị đảo | Bỏ `--headless` không còn hiện cửa sổ | `play.py` tự gán `--viz kit` |

> **Bài học:** nâng cấp framework lớn thì thứ vỡ không phải cú pháp, mà là **quy ước ngầm**: thứ tự quaternion, kiểu dữ liệu trả về, mặc định bị đảo. Cờ cũ vẫn nhận, không báo lỗi, chỉ là hết tác dụng. Loại này tệ hơn API gãy, vì API gãy thì nổ ngay còn cái này im lặng làm sai. Cách phát hiện: so hành vi của checkpoint cũ trước và sau khi nâng cấp.

### Nhóm B. rsl-rl dưới 4.0 lên 5.0.1

| Đổi cái gì | Hậu quả | Sửa thế nào |
|---|---|---|
| Tách `ActorCritic` thành `actor` + `critic` | Config cũ không dựng được | `RslRlMLPModelCfg` cho từng cái |
| `alg.actor_critic` → `alg.get_policy()` | `play.py` gãy | API mới, có nhánh dự phòng |
| Checkpoint tách `actor_state_dict` + `critic_state_dict` | 488 file không nạp được | `convert_checkpoint_rslrl5.py` |

> **Bài học:** đổi định dạng lưu trữ nguy hiểm hơn đổi API, vì nó làm **dữ liệu cũ mất giá trị**. Nhưng thường vẫn cứu được. Ở đây kiến trúc không đổi, chỉ đổi tên khóa, và thí nghiệm đối chứng đã chứng minh trọng số còn nguyên (5.2).

### Nhóm C. Code mang dấu vết máy khác

Project vốn chạy trên workstation `tatung-HP-Z4-G4`.

| Chỗ sai | Thực tế trên máy này |
|---|---|
| `num_envs=512` | Đo lại được 4096, nhanh gấp 2.9 lần |
| `${HOME}/IsaacLab/isaac-sim/python.sh` | Không tồn tại (cài bằng pip) |
| `conda activate isaaclab` | Env tên `isaacsim` |
| `python3.11/site-packages` | Env chạy Python 3.12 |
| `run.sh` / `run_direct.sh` / `isaaclab.sh` trỏ đường dẫn chết | Viết lại, gom về `run.sh` |

> **Bài học:** số nào không tự đo trên máy mình thì đừng tin. `num_envs=512` không sai, nó đúng cho máy khác. Hằng số nào ảnh hưởng hiệu năng mà không kèm phép đo thì hãy nghi ngờ. (Áp dụng ngay: `num_envs=256` trong `official_env.py` cũng chưa được đo.)

### Nhóm D. Cấu hình chưa bao giờ được đo

| Chỗ | Vấn đề | Kết quả |
|---|---|---|
| Buffer PhysX mặc định | Thiết kế cho bài manipulation (tay máy gắp vật) với hàng nghìn tiếp xúc | Tiết kiệm 1650 MiB |
| `isaac-perf` ghi thẳng `platform_profile` | Ghi đè lẫn nhau với `power-profiles-daemon` | Đi qua `powerprofilesctl`, state ở `/var/lib` |
| `from .ui_extension_example import *` | `import transformer_nam` gãy ngoài SimulationApp | Bỏ dòng import (boilerplate không dùng) |
| `outputs/` 15 MB rác hydra, run rỗng | — | Xóa |

> **Bài học:** giá trị mặc định là phỏng đoán của người viết thư viện cho trường hợp trung bình, không phải giá trị tối ưu cho bài của bạn.

### Nhóm E. Mở cửa sổ khi play, và hai lỗi do chính đợt sửa gây ra

| # | Lỗi | Vì sao | Sửa |
|---|---|---|---|
| 1 | Play tua quá nhanh | Không ai đặt `--real-time` | `play.py` tự bật `real_time` |
| 2 | Render có thể rơi xuống iGPU Intel | `prime-select` ở chế độ `on-demand` | Thêm `__NV_PRIME_RENDER_OFFLOAD`, `__VK_LAYER_NV_optimus`, `__GLX_VENDOR_LIBRARY_NAME` vào `run.sh` |
| 3 | `TwL`/`TwR` đọc nhầm khớp | `play.py` viết cứng index, USD lại xếp xen kẽ | Tra chỉ số theo tên khớp |
| 4 | **Tự gây:** `--viz none` bị ghi đè thành mở cửa sổ | `AppLauncher._parse_visualizer_csv` trả `None` cho cả "không truyền" lẫn "truyền `none`" | Dùng cờ `visualizer_explicit`, chỉ bật khi cờ được gõ thật. Bắt được nhờ chạy thử 5 tổ hợp cờ |
| 5 | **Tự gây:** `--rendering_mode performance` làm viewport đen | Thêm vào vì "GPU 6 GB nên giảm tải", **không kiểm bằng mắt** | Gỡ hẳn. Bắt được nhờ user so ảnh viewport với GIF mẫu |

> **Bài học 1 — "máy treo" và "đang chạy nhưng im lặng" trông giống hệt nhau.** Đã tắt nhầm hai lần ở giây 74 và 80, trong khi đích là giây 124. Trước khi kết luận treo, phải biết mốc bình thường.
>
> **Bài học 2 — chỉ số vào mảng thì tra theo tên.** Thứ tự khớp do file USD quyết định, không phải do code. Manh mối vụ `TwL`/`TwR` là **giới hạn khớp**: số báo ±25° trong khi khớp xoay cho phép ±180°. Số không bao giờ chạm gần giới hạn của khớp nó tự nhận là dấu hiệu đọc nhầm khớp.
>
> **Bài học 3 — đồ họa phải kiểm bằng mắt, không kiểm bằng log.** Sim vẫn chạy, log vẫn đẹp, nhìn từ terminal không thể biết viewport đang đen. "Tối ưu cho máy yếu" nghe hợp lý nên rất dễ thêm vào mà không kiểm chứng.

### Số đo trước / sau

| | Trước | Sau |
|---|---|---|
| `num_envs` mặc định (env cũ) | 512 | **4096** |
| Tốc độ | 14 288 steps/s | **41 304 steps/s** |
| VRAM ở mức đó | 3547 MiB | **1897 MiB** |
| RAM thấp nhất khi chạy | 2.9 Gi (ở 8192) | **6.6 Gi** |
| 1500 vòng | khoảng 3 giờ | **khoảng 1 giờ** |
| Play mở cửa sổ | không mở được | **có**: 124 s lần đầu, 23 s lần sau |

### Đã kiểm chứng bằng chạy thật (2026-08-03)

| Kiểm tra | Kết quả |
|---|---|
| Train 2 task × 2 mức `num_envs` | 0 lỗi, 0 overflow PhysX |
| Resume `model_1499_rslrl5.pt` | `iter` khôi phục đúng, ep_len 23.9 → 69.7 |
| Train từ đầu (đối chứng) | ep_len 18.5 → 18.0 |
| Play 10DOF có cửa sổ, `model_1499` | Robot hiện trên lưới sàn, chạy tới step 4246 |
| Play 6DOF có cửa sổ, `model_999` | Chạy tới step 8618, 0 lỗi |
| Chỉ số khớp xoay | `Twistleft=[4] Twistright=[5]`, khớp bảng Isaac Sim |
| 5 tổ hợp cờ (trần / `--headless` / `--viz none` / `--viz kit` / `--viz rerun`) | 5/5 đạt |
| Train sau khi sửa play | `--max_iterations 3` exit 0, ep_len 15–18 |

---

## 10. Dùng git để soát lại

Repo git nằm ở `Transform_bipedal_todai/`, **một cấp trên** `isaac_rl/`.

```bash
cd ~/Documents/projects/Transformer/Transform_bipedal_todai
git status          # đang sửa gì
git diff            # sửa cụ thể ra sao
```

`logs/`, `outputs/`, `*.pt`, `*.onnx`, `events.out.tfevents.*` đều đã trong `.gitignore`.

Thói quen khi chỉnh thưởng:

```bash
git diff                    # soát trước khi train, bắt được lỗi gõ nhầm số
git stash                   # thử một hướng, không thích thì bỏ
git stash pop               # lấy lại
git add -p && git commit    # commit từng phần
```

Commit message nên ghi **con số**, ví dụ `feet_h 2 -> 3, velocity 1.2 -> 1.5`. Vài tuần sau nhìn lại vẫn hiểu. Chi tiết git xem `docs/git.md`.

---

## 11. Tra cứu chéo

| Cần gì | Xem ở đâu |
|---|---|
| Thuật toán, PPO, mẫu config rsl_rl, reward, sim-to-real, việc phải làm, lỗi env 10DOF cũ | `docs/ALGO.md` |
| Kế hoạch đang làm / sửa sau review | `PLAN.md`, `FIX_AFTER_DIFF.md` |
| Run nào chạy được với task nào | `isaac_rl/logs/README-runs.md` |
| Cấu trúc thư mục | `docs/project_structure.md` |
| Git | `docs/git.md` |
| Bản sao code trước khi sửa | `isaac_rl/.backup_truoc_khi_sua_20260803_010207/` |
