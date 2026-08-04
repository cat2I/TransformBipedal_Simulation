# Sổ tay vận hành — Transformer bipedal

Lập 2026-08-03. Dành cho vòng lặp hằng ngày: **sửa hàm thưởng → train → play thử**.

Tài liệu này nói về **quy trình làm việc**. Còn tài liệu về **máy** (cài đặt, tối ưu
CPU/RAM, số đo `num_envs`) nằm ở `CAI-DAT-ISAACLAB.md` — cùng thư mục gốc repo với
file này, không phải ở `~`.

---

## 0. Ba mươi giây để chạy

**Luôn đứng ở `transformer_nam/` trước.** `run.sh` nằm trong đó, không phải gốc repo —
đứng sai chỗ sẽ báo `bash: ./run.sh: No such file or directory`.

```bash
cd ~/Documents/projects/Transformer/Transform_bipedal_tovinh/Transform_bipedal/transformer_nam
```

**Play — xem robot cử động trong cửa sổ Isaac Sim** (policy tốt nhất hiện có):

```bash
./run.sh scripts/rsl_rl/play.py --task Transformer-Walk10DOF-Direct-v0 --num_envs 1 \
    --checkpoint "$PWD/logs/rsl_rl/transformer_walk/2026-07-23_15-23-03/model_1499_rslrl5.pt"
```

Không cần thêm cờ nào — cửa sổ tự mở, tốc độ tự đúng thật. Lần đầu sau khi bật máy mất
~2 phút, những lần sau ~23 giây. Chi tiết ở Bước 6.

**Train thử 3 vòng:**

```bash
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 --headless --max_iterations 3
```

**Không cần `conda activate`.** `run.sh` tự kích hoạt env `isaacsim`, tự đặt
`PYTHONPATH`, tự đặt biến môi trường cho GPU 6GB, và tự dùng `isaac-run` nếu có.
Mọi lệnh dưới đây đều chạy từ thư mục `transformer_nam`.

> **Khác nhau cơ bản giữa train và play:**
> train luôn `--headless` (nhanh, tiết kiệm VRAM); play **không** truyền gì cả thì tự
> có cửa sổ. Đừng thêm `--rendering_mode` vào play — xem `CAI-DAT-ISAACLAB.md` mục 10.

`run_direct.sh` và `isaaclab.sh` chỉ chuyển tiếp sang `run.sh` — dùng cái nào cũng như nhau.

---

## 1. Vòng lặp chính

### Bước 1 — Sửa hàm thưởng

Tất cả nằm trong **một file duy nhất** ứng với task bạn dùng:

| Task | File |
|---|---|
| `Transformer-Walk10DOF-Direct-v0` (60 obs / 10 act) | `transformer_walk10dof_env.py` |
| `Transformer-Walk10DOF6-Direct-v0` (44 obs / 6 act) | `transformer_walk10dof6_env.py` |

Đường dẫn đầy đủ:
`source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/<file>`

Trong file đó:

| Muốn sửa gì | Ở dòng (bản 10DOF) | (bản 6DOF) |
|---|---|---|
| Tỉ trọng 7 thành phần thưởng | **75–77** `weights = {...}` | 67–69 |
| Cách cộng thành tổng | **345** `_get_rewards()` | 344 |
| Điều kiện ngã / kết thúc sớm | **386** `_get_dones()` | 385 |
| Nội dung quan sát đưa vào policy | 262 `_get_observations()` | 256 |
| Độ dài tập, số env | 44, 106 | 48, 93 |
| Công thức từng thành phần | **477–632** (các hàm `@torch.jit.script`) | tương tự |

Bảy vị trí trong `weights` theo đúng thứ tự:

```python
weights = {
    #        orient  height  position  sig_extra  feet_h  velocity  deviation
    "walk": [   1,      1,       1,         0,       2,      1.2,       1    ],
}
```

| # | Tên | Hàm tính | Ý nghĩa |
|---|---|---|---|
| 0 | orientation | `orientation_reward` (514) | thân giữ thẳng, phạt nghiêng |
| 1 | height | `height_reward` (558) | giữ hông ở **0.43 m** |
| 2 | position | `joint_position_reward` (575) | khớp gần tư thế gốc |
| 3 | sig_extra | `sigmoid_extra` (609) | **đang tắt** (trọng số 0) |
| 4 | feet_h | `feet_height_reward` (619) | nhấc chân đủ cao khi bước |
| 5 | velocity | `velocity_reward` (588) | đi tới đúng hướng |
| 6 | deviation | `deviation_reward` (533) | không lệch khỏi đường thẳng |

#### ⚠ Ba cái bẫy phải biết trước khi sửa

**Bẫy 1 — Trọng số là TỈ LỆ, không phải hệ số.**
Dòng 361 có:
```python
w = self.weights / torch.sum(self.weights, dim=1, keepdim=True)
```
Bảy số được chia cho tổng của chúng. Nên tăng `feet_h` từ `2` lên `4` **không** làm
nó mạnh gấp đôi — nó làm **mọi thành phần khác yếu đi**. Tổng hiện tại là
`1+1+1+0+2+1.2+1 = 7.2`, nên `feet_h` đang chiếm `2/7.2 = 27.8%`.

Muốn tăng riêng một thành phần mà giữ nguyên các thành phần khác thì phải **giảm
các số còn lại**, hoặc sửa thẳng công thức trong hàm thưởng tương ứng.

**Bẫy 2 — Các hàm thưởng có `@torch.jit.script`, kiểu dữ liệu bị soi rất nghiêm.**
IsaacLab 3.0 trả về `ProxyArray` chứ không phải `torch.Tensor` khi bạn đọc
`robot.data.*`, và **ProxyArray không lọt qua được jit**. Vì vậy mọi chỗ đọc dữ liệu
phải bọc `as_torch(...)`:

```python
# ĐÚNG
robot_root_pos = as_torch(self.robot.data.root_pos_w)
air_time       = as_torch(self.scene.sensors["contact"].data.current_air_time)

# SAI — sẽ nổ khi truyền vào hàm @torch.jit.script
robot_root_pos = self.robot.data.root_pos_w
```

`as_torch` nhập từ `._lab3_compat`, đã có sẵn ở đầu mỗi file env.

Ngoài ra jit **không nhận** `Optional`, `dict`, hay kiểu động — tham số nào không
phải Tensor thì phải chú thích rõ (`action: str`, `target_h: float`).

**Bẫy 3 — Đổi số chiều quan sát/hành động là mất hết checkpoint cũ.**
Nếu bạn sửa `_get_observations()` làm đổi `observation_space` (dòng 50), hoặc đổi
`action_space` (dòng 64), thì mọi checkpoint đã train **không nạp lại được** —
mạng nơ-ron sai kích thước. Đây không phải lỗi. Chấp nhận train lại từ đầu, hoặc
giữ nguyên số chiều.

Sửa **công thức thưởng** thì không sao — checkpoint vẫn nạp được, chỉ là chính sách
cũ được đánh giá theo thước đo mới.

---

### Bước 2 — Kiểm tra cú pháp (2 giây, tiết kiệm 40 giây)

Isaac Sim mất ~40 giây để khởi động. Đừng để nó khởi động xong mới báo lỗi thụt lề.

```bash
python -m py_compile source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/*.py && echo OK
```

Soát lại thay đổi của mình bằng git (xem mục 5):
```bash
cd .. && git diff --stat && cd transformer_nam
```

---

### Bước 3 — Chạy thử 3 vòng trước khi cam kết

```bash
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 \
    --headless --max_iterations 3
```

**Nhìn gì để biết ổn:**

| Dấu hiệu | Nghĩa là |
|---|---|
| `Mean reward` ra số bình thường | thưởng tính được |
| `Mean reward: nan` | công thức chia cho 0 hoặc `log` số âm — sửa ngay |
| `Mean episode length` > 10 | robot không ngã tức thì |
| Không có `Traceback` | không lỗi kiểu dữ liệu |
| Không có dòng nào chứa `overflow` | buffer PhysX đủ dùng |

Nếu thưởng ra `nan` mà không thấy lỗi, tăng dần `--max_iterations` để tìm vòng nào
bắt đầu hỏng.

---

### Bước 4 — Train thật

**Train mới từ đầu** (dùng khi đã đổi hàm thưởng đáng kể — chính sách cũ không còn hợp):

```bash
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 \
    --headless --max_iterations 1500
```

**Train tiếp từ checkpoint có sẵn** (dùng khi chỉ tinh chỉnh nhẹ):

```bash
./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 \
    --headless --max_iterations 1500 \
    --resume --load_run 2026-07-23_15-23-03 --checkpoint model_1499_rslrl5.pt
```

**Không cần truyền `--num_envs`** — mặc định đã là **4096**, con số đo trên chính
máy này. Muốn nhanh thêm ~10% thì đóng hết ứng dụng rồi thêm `--num_envs 8192`.
Trên 8192 không nhanh thêm chút nào, đừng nâng nữa.

**Ước lượng thời gian** ở tốc độ đo được (41 304 steps/s, 4096 env):
khoảng **2.3 giây mỗi vòng** → 1500 vòng ≈ **1 giờ**.

Checkpoint tự lưu mỗi 50 vòng vào
`logs/rsl_rl/transformer_walk/<ngày-giờ>/model_<N>.pt`.

> **Lưu ý khi resume:** vòng đầu tiên `mean_episode_length` sẽ **tụt mạnh** rồi mới
> leo lại (ví dụ 83 → 24 → 70 sau 10 vòng). Bình thường, do `init_at_random_ep_len`
> và Adam khởi động lại. **Đừng tưởng là hỏng.** Nếu sau 20 vòng vẫn không leo lại
> thì mới đáng nghi.

---

### Bước 5 — Theo dõi bằng tensorboard

Mở **terminal thứ hai**:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacsim
cd ~/Documents/projects/Transformer/Transform_bipedal_tovinh/Transform_bipedal/transformer_nam
tensorboard --logdir=logs/rsl_rl/transformer_walk --port=6006
```

Mở trình duyệt vào `http://localhost:6006`.

**Đường quan trọng nhất: `Train/mean_episode_length`.**

Tập tối đa = `episode_length_s 10.0` ÷ (`dt 0.005` × `decimation 10`) = **200 bước**.

| Giá trị | Nghĩa |
|---|---|
| ~18 | policy ngẫu nhiên, chưa học được gì |
| 83 | **mức tốt nhất hiện có** — robot ngã ở giây 4.2/10 |
| 200 | sống trọn 10 giây — mục tiêu |

`Train/mean_reward` chỉ để so hai lần chạy **cùng hàm thưởng**. Đổi trọng số rồi thì
số reward không so được với lần trước nữa — nhưng `mean_episode_length` thì **vẫn so
được**, vì nó đo hành vi thật chứ không đo thước đo. Khi chỉnh thưởng, hãy nhìn cột
này.

---

### Bước 6 — Play thử

Nhớ đứng ở `transformer_nam/` — `run.sh` nằm trong đó, không phải ở gốc repo.

```bash
cd ~/Documents/projects/Transformer/Transform_bipedal_tovinh/Transform_bipedal/transformer_nam
./run.sh scripts/rsl_rl/play.py --task Transformer-Walk10DOF-Direct-v0 --num_envs 1 \
    --checkpoint "$PWD/logs/rsl_rl/transformer_walk/2026-07-23_15-23-03/model_1499_rslrl5.pt"
```

Lệnh này **tự mở cửa sổ Isaac Sim** và chạy đúng tốc độ thật — không cần thêm cờ nào.

> **Lần đầu mất ~2 phút và trông y như máy treo. Đừng tắt.** Những lần sau chỉ còn
> ~23 giây (số đo thật ở `CAI-DAT-ISAACLAB.md` mục 7). Biết là xong khi terminal in
> dòng `[    0] roll=...`. Nếu Ubuntu hiện *"not responding"* thì xem bảng lỗi mục 3.

> ### ⚠ Khác biệt dễ vấp nhất
> | Script | `--checkpoint` nhận |
> |---|---|
> | `train.py --resume` | **tên file**: `model_1499_rslrl5.pt` |
> | `play.py` | **đường dẫn đầy đủ**: `$PWD/logs/.../model_1499_rslrl5.pt` |
>
> Đưa tên file trần cho `play.py` sẽ báo `FileNotFoundError`. Không phải lỗi project.

**Muốn chạy ngầm** (chỉ xem log, không cửa sổ) thì thêm `--headless`.

> #### Vì sao phải sửa code mới mở được cửa sổ — IsaacLab 3.0 đảo ngược mặc định
>
> | | IsaacLab 2.x | IsaacLab 3.0 (đang cài) |
> |---|---|---|
> | Mặc định | **có cửa sổ** | **headless** |
> | Tắt cửa sổ | `--headless` | (mặc định) hoặc `--viz none` |
> | Bật cửa sổ | (mặc định) | **`--viz kit`** |
>
> `--headless` giờ chỉ là cờ deprecated, nên **bỏ nó đi không còn làm hiện cửa sổ**.
> Thủ phạm nằm ở `~/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py`, hàm
> `_resolve_headless_settings`: không chọn visualizer nào thì nó tự ép
> `self._headless = True`.
>
> `play.py` đã được sửa để tự gán `--viz kit` khi bạn không nói gì, nên lệnh trên
> chạy trần vẫn ra cửa sổ. Muốn tự chọn thì truyền `--viz <tên>` — lúc đó `play.py`
> không can thiệp nữa, và `--real-time` cũng không tự bật (chạy nhanh tối đa).

**Đọc dòng log của play:**

```
[ 1348] roll=+0.172 pitch=-0.632 gz=-0.290 h=0.397m | TwL=+23.7° TwR=+5.1° | air L=0.45s R=0.00s contact L=n R=Y
```

| Trường | Nghĩa | Muốn thấy gì |
|---|---|---|
| `h=` | độ cao hông | quanh **0.43 m**; xuống dưới 0.1 là ngã |
| `roll` / `pitch` | nghiêng (radian) | gần 0; vượt **0.95** là kết thúc tập |
| `TwL` / `TwR` | góc xoay hai chân | dao động đều = đang bước |
| `air L/R` | thời gian chân treo | **so le nhau** = bước thật, không phải nhảy cóc |
| `contact L/R` | chạm đất | luân phiên Y/n |

Cả hai chân `air` cùng tăng = robot đang nhảy, không phải đi. Cả hai cùng 0 = đang
đứng yên.

> ### ⚠ Log sinh TRƯỚC 03-08-2026 có hai cột `TwL`/`TwR` sai khớp
>
> Bản `play.py` cũ giả định thứ tự khớp là "trái hết rồi tới phải" nên viết cứng
> index 2 và 7. Thứ tự thật lại **xen kẽ**:
>
> ```
> 0 Bubleft  1 Bubright  2 Hipleft  3 Hipright  4 Twistleft
> 5 Twistright 6 Kneeleft 7 Kneeright 8 Footleft 9 Footright
> ```
>
> Nên index 2 là **Hipleft** và index 7 là **Kneeright**, không phải hai khớp xoay.
> Log cũ in `TwL≈+24°` đứng yên suốt chính là góc hông (giới hạn ±37°), không phải
> góc xoay (giới hạn ±180°). **Đừng dùng log cũ để đánh giá chuyển động xoay.**
>
> Đã sửa: `play.py` giờ tra chỉ số theo tên khớp và in ra
> `Khớp xoay: Twistleft=[4] Twistright=[5]` lúc khởi động để đối chiếu.

---

### Bước 7 — Xuất policy cho robot thật

`play.py` **tự động** xuất khi chạy, vào:

```
logs/rsl_rl/transformer_walk/<run>/exported/policy.pt     ← TorchScript
logs/rsl_rl/transformer_walk/<run>/exported/policy.onnx   ← ONNX
```

Hai file này là mạng đã đóng gói, không cần Isaac Sim để chạy.

---

## 2. Bảng lệnh tra nhanh

Tất cả chạy từ `transformer_nam/`.

| Việc | Lệnh |
|---|---|
| Thử nhanh 3 vòng | `./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 --headless --max_iterations 3` |
| Train mới | `./run.sh scripts/rsl_rl/train.py --task Transformer-Walk10DOF-Direct-v0 --headless --max_iterations 1500` |
| Train tiếp | thêm `--resume --load_run <thư-mục> --checkpoint <tên-file>` |
| Train task 6DOF | đổi thành `--task Transformer-Walk10DOF6-Direct-v0` |
| **Play (có cửa sổ)** | `./run.sh scripts/rsl_rl/play.py --task Transformer-Walk10DOF-Direct-v0 --num_envs 1 --checkpoint "$PWD/logs/rsl_rl/transformer_walk/2026-07-23_15-23-03/model_1499_rslrl5.pt"` |
| Play policy cũ (44/6) | như trên, đổi `--task Transformer-Walk10DOF6-Direct-v0` và trỏ checkpoint 44/6 |
| Play không cửa sổ | thêm `--headless` |
| Tensorboard | `tensorboard --logdir=logs/rsl_rl/transformer_walk --port=6006` |
| Chuyển checkpoint cũ sang định dạng mới | `./run.sh scripts/convert_checkpoint_rslrl5.py logs/rsl_rl/transformer_walk/<run>` |
| Kiểm tra cú pháp | `python -m py_compile source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/*.py` |
| Xem CPU/RAM/swap | `isaac-perf status` |
| Xem run nào dùng task nào | `cat logs/README-runs.md` |
| Soát thay đổi của mình | `cd .. && git diff` |

---

## 3. Khi gặp lỗi — tra ở đây

Mười một lỗi này đã gặp thật và đã sửa. Nếu gặp lại, tra bảng trước khi tìm chỗ khác.

| Thông báo lỗi | Nguyên nhân gốc | Xử lý |
|---|---|---|
| `bash: ./run.sh: No such file or directory` | Đang đứng ở gốc repo, mà `run.sh` nằm trong `transformer_nam/` | `cd transformer_nam` rồi chạy lại |
| Chạy được, in log bình thường, **nhưng không mở cửa sổ** | IsaacLab 3.0 đảo mặc định: không có `--viz` là headless. Bỏ `--headless` không còn đủ nữa | Đã sửa trong `play.py` — nó tự thêm `--viz kit`. Script khác thì tự truyền cờ đó |
| Cửa sổ hiện *"Isaac Lab is not responding"* lặp đi lặp lại | GNOME ping cửa sổ, quá `check-alive-timeout` (mặc định **5 giây**) là báo treo — mà Kit đứng lâu hơn thế liên tục lúc khởi động | `gsettings set org.gnome.mutter check-alive-timeout 0`<br>Hoàn tác: `gsettings reset org.gnome.mutter check-alive-timeout` |
| Cửa sổ mở, log chạy đều, **nhưng khung 3D đen thui** (không thấy cả lưới sàn) | Đang chạy kèm `--rendering_mode performance`. Preset đó tắt quá nhiều tính năng RTX nên không dựng được gì | Bỏ cờ đó đi, để mặc định. Chi tiết ở `CAI-DAT-ISAACLAB.md` mục 10 |
| `ImportError: cannot import name 'as_torch'` | Ai đó xoá hàm `as_torch` khỏi `_lab3_compat.py` trong khi 7 file env vẫn import | Khôi phục hàm, **đừng** xoá dòng import |
| Lỗi kiểu dữ liệu lạ trong hàm có `@torch.jit.script` | `ProxyArray` của IsaacLab 3.0 lọt vào jit | Bọc `as_torch(...)` ở chỗ đọc `robot.data` |
| `KeyError: 'actor_state_dict'` | Checkpoint train bằng rsl_rl < 4.0, bản 5.0.1 đòi định dạng mới | `./run.sh scripts/convert_checkpoint_rslrl5.py <thư-mục-run>` |
| `'PPO' object has no attribute 'actor_critic'` | Code viết cho rsl_rl < 4.0 | Dùng `runner.alg.get_policy()` |
| `IndexError: index 60 is out of bounds` | Code hard-code cho task 62 chiều, đang chạy task 60 chiều | Đọc số chiều từ `cfg` thay vì viết cứng |
| `FileNotFoundError: Unable to find the file: model_99.pt` | `play.py` cần **đường dẫn đầy đủ** | Thêm `$PWD/logs/...` phía trước |
| `ModuleNotFoundError: No module named 'omni.ext'` | `omni.ext` chỉ nạp được **sau** khi SimulationApp khởi động | Đừng import ở mức module; train/play không bị lỗi này |
| Log có chữ `overflow` / `capacity` | Buffer PhysX đặt thấp quá cho scene mới | PhysX **tự in con số nó cần** — nâng đúng tham số đó lên nấc 2ⁿ kế tiếp, đừng nâng cả bảng |

**Robot ngã ngay lập tức sau khi sửa thưởng** (`mean_episode_length` ~15 và không lên):
kiểm tra `_get_dones()` dòng 386 — ngưỡng hiện tại là độ cao hông `< 0.1 m` hoặc
`|roll|`/`|pitch| > 0.95 rad`. Sửa thưởng làm robot khom xuống dưới 0.1 m là bị
kết thúc tập tức thì dù chưa thật sự ngã.

---

## 4. Những gì đã sửa và vì sao — để hiểu hệ thống

Mười bốn thứ, nhóm theo **nguyên nhân gốc** thay vì thứ tự thời gian, vì nhóm như
vậy mới rút ra bài học dùng lại được.

### Nhóm A — IsaacLab 2.x lên 3.0 đổi API (4 mục)

| Đổi cái gì | Hậu quả | Sửa thế nào |
|---|---|---|
| `robot.data.*` trả `ProxyArray` thay vì `torch.Tensor` | không lọt qua `@torch.jit.script` | bọc `as_torch()` ở **56 chỗ** trong 7 file |
| Quaternion đổi `(w,x,y,z)` → `(x,y,z,w)` | góc nghiêng tính sai hoàn toàn | sửa `quaternion_to_euler`, thêm cờ bù trong `_lab3_compat.py` |
| Sensor IMU mất `quat_w` (chuyển sang sensor `Pva`) | không đọc được hướng thân | hàm bù `imu_quat_w()` |
| `PhysxCfg` chuyển sang package `isaaclab_physx` | import cũ gãy | `from isaaclab_physx.physics import PhysxCfg` |

> **Bài học:** nâng cấp framework lớn thì thứ vỡ không phải cú pháp mà là **quy ước
> ngầm** — thứ tự thành phần quaternion, kiểu dữ liệu trả về. Loại này trình biên
> dịch không bắt được, chỉ lộ ra khi robot cư xử kỳ lạ. Cách phát hiện: so hành vi
> của checkpoint cũ trước và sau khi nâng cấp.

### Nhóm B — rsl-rl dưới 4.0 lên 5.0.1 đổi API (3 mục)

| Đổi cái gì | Hậu quả | Sửa thế nào |
|---|---|---|
| Tách `ActorCritic` thành `actor` + `critic` rời | config cũ không dựng được | `RslRlMLPModelCfg` cho từng cái |
| `alg.actor_critic` → `alg.get_policy()` | `play.py` gãy | dùng API mới, có nhánh dự phòng |
| Checkpoint: một `model_state_dict` → tách `actor_state_dict` + `critic_state_dict` | 488 file cũ không nạp được | viết `convert_checkpoint_rslrl5.py` |

> **Bài học:** đổi định dạng lưu trữ nguy hiểm hơn đổi API, vì nó làm **dữ liệu cũ
> mất giá trị** chứ không chỉ làm code gãy. Nhưng thường vẫn cứu được: ở đây kiến
> trúc mạng không đổi, chỉ đổi tên khoá, nên chuyển đổi là thuần đổi tên. Đã kiểm
> chứng bằng thí nghiệm đối chứng: resume đạt ep_len 69.7 sau 9 vòng, còn train từ
> đầu chỉ 18.0 — chứng tỏ trọng số cũ còn nguyên giá trị.

### Nhóm C — Code mang dấu vết máy khác (4 mục)

Project vốn chạy trên workstation `tatung-HP-Z4-G4`. Chuyển sang laptop thì:

| Chỗ sai | Thực tế trên máy này |
|---|---|
| `num_envs=512` | đo lại được **4096** — nhanh gấp 2.9 lần |
| `${HOME}/IsaacLab/isaac-sim/python.sh` | không tồn tại (cài bằng pip, không phải bản đóng gói) |
| `conda activate isaaclab` | env tên là `isaacsim` |
| `python3.11/site-packages` | env chạy Python 3.12 |

> **Bài học:** số nào không tự đo trên máy mình thì đừng tin. `num_envs=512` không
> sai — nó đúng cho máy khác. Cách phát hiện: bất cứ hằng số nào ảnh hưởng hiệu năng
> mà không kèm phép đo, hãy nghi ngờ.

### Nhóm D — Cấu hình chưa bao giờ được đo (3 mục)

| Chỗ | Vấn đề | Kết quả sau khi sửa |
|---|---|---|
| Buffer PhysX để mặc định | mặc định sized cho manipulation hàng nghìn tiếp xúc; robot này chỉ chạm đất ở `Foot.*` | tiết kiệm **1650 MiB** |
| `isaac-perf` ghi thẳng `platform_profile` | đánh nhau với `power-profiles-daemon`, làm máy âm thầm tụt xuống `balanced` | đi qua `powerprofilesctl`, state lưu ở `/var/lib` |
| `from .ui_extension_example import *` ở `__init__.py` | `import transformer_nam` gãy ngoài SimulationApp | bỏ dòng import (boilerplate template không dùng) |

> **Bài học:** giá trị mặc định là **phỏng đoán của người viết thư viện về trường
> hợp trung bình**, không phải giá trị tối ưu cho bài của bạn. Ở đây riêng
> `gpu_max_soft_body_contacts` và `gpu_max_particle_contacts` giữ chỗ cho 2²⁰ tiếp
> xúc mỗi loại, trong khi scene **không có soft body lẫn particle nào**.

### Nhóm E — Mở được cửa sổ Isaac Sim khi play (4 mục)

Triệu chứng ban đầu: `play.py` chạy ngon, in đủ log, robot cử động — mà **không có cửa
sổ nào**.

| Đổi cái gì | Hậu quả | Sửa thế nào |
|---|---|---|
| IsaacLab 3.0 **đảo mặc định**: không có `--viz` là headless | bỏ `--headless` không còn làm hiện cửa sổ; tài liệu cũ chỉ sai | `play.py` tự gán `--viz kit` khi người dùng không nói gì |
| Play chạy nhanh tối đa | nhìn không kịp | cùng nhánh trên bật `real_time` |
| GNOME `check-alive-timeout` = 5 giây | hộp thoại *not responding* hiện lại chục lần khi Kit khởi động (mất 2 phút) | đặt về 0 (`CAI-DAT-ISAACLAB.md` mục 9) |
| `play.py` in `TwL`/`TwR` nhưng đọc index 2/7 | hai cột đó thật ra là **Hipleft/Kneeright**, không phải khớp xoay (ở 4/5) | tra chỉ số **theo tên khớp**, không viết cứng số |

> **Bài học 1 — nâng cấp lớn thì thứ vỡ đau nhất là mặc định bị đảo ngược.** Cờ cũ vẫn
> còn, vẫn nhận, không báo lỗi — chỉ là hết tác dụng. Tệ hơn API gãy, vì API gãy thì nổ
> ngay còn cái này im lặng làm sai.
>
> **Bài học 2 — "máy treo" và "máy đang chạy nhưng không nói gì" nhìn giống hệt nhau.**
> Đã tắt nhầm hai lần ở giây 74 và 80 trong khi đích là giây 124. Trước khi kết luận
> treo, phải biết mốc bình thường là bao nhiêu.
>
> **Bài học 3 — chỉ số vào mảng thì tra theo tên.** Thứ tự khớp do file USD quyết định,
> không phải do code. Manh mối để lộ ra vụ TwL/TwR là **giới hạn khớp**: số báo ±25°
> trong khi khớp xoay cho phép ±180°, còn khớp hông thì đúng ±37°.

### Số đo trước / sau

| | Trước | Sau |
|---|---|---|
| `num_envs` mặc định | 512 | **4096** |
| Tốc độ | 14 288 steps/s | **41 304 steps/s** |
| VRAM ở mức đó | 3547 MiB | **1897 MiB** |
| RAM thấp nhất khi chạy | 2.9 Gi (ở 8192) | **6.6 Gi** |
| Thời gian 1500 vòng | ~3 giờ | **~1 giờ** |
| Play mở cửa sổ | *không mở được* | **có** — 124s lần đầu, 23s lần sau |

---

## 5. Dùng git để soát lại

Repo git nằm ở **`Transform_bipedal/`** — một cấp **trên** `transformer_nam/`.

```bash
cd ~/Documents/projects/Transformer/Transform_bipedal_tovinh/Transform_bipedal
git status          # xem đang sửa gì
git diff            # xem sửa cụ thể ra sao
```

`logs/`, `outputs/`, `*.pt`, `*.onnx`, `events.out.tfevents.*` đều đã trong
`.gitignore` — commit không sợ nặng.

**Thói quen nên có khi chỉnh thưởng:**

```bash
git diff                    # soát trước khi train — bắt được lỗi gõ nhầm số
git stash                   # thử một hướng, không thích thì bỏ
git stash pop               # lấy lại
git add -p && git commit    # commit từng phần, ghi rõ đổi trọng số nào
```

Commit message nên ghi **con số**, ví dụ `feet_h 2 -> 3, velocity 1.2 -> 1.5` — vài
tuần sau nhìn lại còn hiểu.

> **Việc nên làm ngay:** `git status` hiện đang có khoảng 20 file sửa đổi chưa commit
> từ hai phiên vừa rồi. Nên commit một lần cho sạch **trước khi** bắt đầu chỉnh
> thưởng, để về sau `git diff` chỉ hiện đúng thay đổi của bạn.

---

## 6. Tra cứu chéo

| Cần gì | Xem ở đâu |
|---|---|
| Cài đặt máy, tối ưu CPU/RAM/swap, số đo `num_envs` | `CAI-DAT-ISAACLAB.md` (gốc repo) |
| Run nào chạy được với task nào (81 run) | `transformer_nam/logs/README-runs.md` |
| Bản sao code trước khi sửa | `transformer_nam/.backup_truoc_khi_sua_20260803_010207/` |
