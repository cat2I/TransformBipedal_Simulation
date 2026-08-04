# Isaac Sim + IsaacLab — hiện trạng và cách dùng

Cập nhật **2026-08-03**. Bản trước của file này viết khi máy còn thiếu IsaacLab;
giờ đã cài đủ nên toàn bộ phần "cài đặt" cũ (Bước 1–5) **không còn đúng**, đã
thay bằng nội dung dưới đây.

---

## 1. Máy đang có gì

| Thành phần | Bản | Ghi chú |
|---|---|---|
| conda env | `isaacsim` | Python 3.12.13 |
| Isaac Sim | 6.0.1.0 | cài bằng pip |
| IsaacLab | 3.0.0-beta2.patch1 | cài editable từ `~/IsaacLab`. **Đảo mặc định headless/GUI so với 2.x** — xem mục 2 |
| rsl-rl-lib | 5.0.1 | API mới, tách actor/critic |
| torch | 2.11.0+cu130 | CUDA nhận GPU |
| warp-lang | 1.13.0 | |
| numpy | 2.3.1 | |
| tensorboard / gymnasium / tqdm / hydra-core | có đủ | |
| transformer_nam | 0.1.0 | cài editable |

**Không thiếu gói nào.** Không cần cài thêm gì. `pip check` báo 2 cảnh báo nhưng
đều vô hại và **cố ý không sửa** — xem mục 8.

Lưu ý: **không có** thư mục `~/IsaacLab/isaac-sim`. Đó là cấu trúc của bản Isaac
Sim đóng gói sẵn; máy này cài bằng pip nên không có `python.sh` hay
`setup_python_env.sh`. Script nào trỏ vào đó là script cũ, đã sửa.

---

## 2. Chạy — cách ngắn nhất

Mở Terminal (`Ctrl+Alt+T`), rồi:

```bash
cd ~/Documents/projects/Transformer/Transform_bipedal_tovinh/Transform_bipedal/transformer_nam
```

### Train ngầm (không hiện cửa sổ — tiết kiệm RAM và VRAM)

```bash
./run.sh scripts/rsl_rl/train.py \
    --task Transformer-Walk10DOF-Direct-v0 \
    --headless --max_iterations 350
```

Không cần truyền `--num_envs` nữa: mặc định đã là **4096**, đo trên chính máy này
(xem mục 5). Con số 512 cũ là của workstation, dùng ở đây phí gần 3 lần tốc độ.

### Train tiếp từ checkpoint có sẵn

```bash
./run.sh scripts/rsl_rl/train.py \
    --task Transformer-Walk10DOF-Direct-v0 --headless --max_iterations 1500 \
    --resume --load_run 2026-07-23_15-23-03 --checkpoint model_1499_rslrl5.pt
```

Lưu ý `--checkpoint` ở chế độ `--resume` nhận **tên file**, còn `play.py` thì cần
**đường dẫn đầy đủ**. Hai script xử lý khác nhau, không phải lỗi.

### Play model đã train

```bash
./run.sh scripts/rsl_rl/play.py \
    --task Transformer-Walk10DOF-Direct-v0 \
    --num_envs 1 \
    --checkpoint "$PWD/logs/rsl_rl/transformer_walk/2026-07-23_15-23-03/model_1499_rslrl5.pt"
```

Lệnh này **tự mở cửa sổ Isaac Sim** và chạy đúng tốc độ thật. Muốn chạy ngầm (chỉ
xem log) thì thêm `--headless`.

> **Lần đầu mở cửa sổ sẽ rất lâu và máy trông như treo.** Kit phải nạp vài trăm
> extension và khởi tạo renderer RTX. Ubuntu sẽ bật hộp thoại *"Isaac Lab 3.0.0 is
> not responding"* — **bấm "Wait", đừng bấm "Force Quit"**. Hộp thoại đó chỉ đo
> event loop đứng vài giây, mà Kit thì đứng liên tục lúc khởi động.
>
> Biết là đã chạy thật khi terminal bắt đầu in dòng `[    0] roll=... pitch=... h=...`.

### Xem biểu đồ train (Terminal thứ 2)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaacsim
cd ~/Documents/projects/Transformer/Transform_bipedal_tovinh/Transform_bipedal/transformer_nam
tensorboard --logdir=logs/rsl_rl/transformer_walk --port=6006
```

Rồi mở trình duyệt vào `http://localhost:6006`.

`run.sh` tự bật conda env, tự đặt `PYTHONPATH`, tự đặt biến môi trường cho GPU
6GB — không phải nhớ gì thêm. `run_direct.sh` và `isaaclab.sh` giờ chỉ chuyển
tiếp sang `run.sh`, dùng cái nào cũng như nhau.

---

## 3. Task đang dùng được

| Task | obs | act |
|---|---|---|
| `Transformer-Walk10DOF-Direct-v0` | 60 | 10 |
| `Transformer-Walk10DOF6-Direct-v0` | 44 | 6 |

`TransformerTwistMarch-v0` **đang bị comment out** trong
`source/transformer_nam/transformer_nam/tasks/direct/transformer_nam/__init__.py`
(dòng 76). Nếu bật lại thì `play.py` vẫn in đủ thông tin twist như trước.

---

## 4. Checkpoint cũ phải chuyển đổi trước khi play

Checkpoint train bằng rsl_rl bản cũ lưu chung một `model_state_dict`. rsl-rl-lib
5.0.1 đòi `actor_state_dict` và `critic_state_dict` tách rời, nên file cũ nạp vào
sẽ báo `KeyError: 'actor_state_dict'`.

Kiến trúc mạng **không đổi**, chỉ khác tên khoá, nên chuyển đổi được:

```bash
./run.sh scripts/convert_checkpoint_rslrl5.py logs/rsl_rl/transformer_walk/<tên-run>
```

File gốc giữ nguyên, bản mới ghi ra `<tên>_rslrl5.pt` bên cạnh.

**Đã chuyển hết 488 checkpoint của toàn bộ 81 run** (2026-08-03). 66 file
`exported/policy.pt` không chuyển vì đó là model JIT đã xuất, không phải
checkpoint train — không cần và không chuyển được.

Optimizer trong file chuyển đổi là bản trống. Play không dùng optimizer nên
không ảnh hưởng; train tiếp từ đó thì Adam chỉ mất vài chục bước để dựng lại.

### Đã kiểm chứng: checkpoint cũ CÒN NGUYÊN GIÁ TRỊ (2026-08-03)

Đây là câu hỏi lớn nhất còn treo: các checkpoint train dưới IsaacLab 2.x, mà 3.0
đổi quy ước quaternion `(w,x,y,z)` → `(x,y,z,w)` và đổi sensor IMU. Nếu lớp bù
trong `_lab3_compat.py` sai thì cả 488 file là rác.

Đã trả lời bằng thí nghiệm đối chứng — hai lần train 10 vòng, cùng
`--num_envs 4096`, chỉ khác một điều là có `--resume` hay không:

| | ep_len đầu → cuối | reward đầu → cuối | action std |
|---|---|---|---|
| Train **từ đầu** | 18.5 → 18.0 *(đứng yên)* | 2.44 → 2.52 | 1.01 |
| **Resume** `model_1499` | 23.9 → **69.7** | 2.93 → **7.60** | 4.04 |

Sau 9 vòng như nhau, bản resume đạt episode length gấp **3.9 lần** và reward gấp
**3 lần**. Policy hỏng thì kết quả phải trùng với cột "từ đầu" — nó khác rất xa.
**Kết luận: shim đúng, dữ liệu cũ dùng được.**

Kiểm tra thêm trực tiếp trên file: `std`, toàn bộ trọng số và `iter` của bản
`_rslrl5.pt` **trùng khít từng chữ số** với file gốc.

Lưu ý khi resume: vòng đầu tiên ep_len tụt xuống ~24 rồi mới leo lại. Bình thường —
do `init_at_random_ep_len=True` trong `train.py` cộng với Adam khởi động lại từ
optimizer trống. Cả hai tự hết sau khoảng 10 vòng. **Đừng tưởng là hỏng.**

### Run nào play được với task nào

Bảng đầy đủ 81 run nằm ở `transformer_nam/logs/README-runs.md`. Tóm tắt:

| obs / act | Task | Số run | Checkpoint |
|---|---|---|---|
| 44 / 6 | `Transformer-Walk10DOF6-Direct-v0` | 45 | 341 |
| 60 / 10 | `Transformer-Walk10DOF-Direct-v0` | 4 | 48 |
| 84 / 10 | *chưa có task khớp* | 4 | 12 |
| 12 / 6 | *chưa có task khớp* | 11 | 39 |
| 14 / 8 | *chưa có task khớp* | 12 | 34 |
| 16 / 8 | *chưa có task khớp* | 2 | 5 |
| 54 / 8 · 56 / 8 · 62 / 10 | *chưa có task khớp* | 3 | 9 |

**49/81 run play được ngay.** 32 run còn lại là thí nghiệm cũ (tháng 2–5) với số
khớp khác; muốn play phải bật lại env tương ứng trong `__init__.py` — không phải
lỗi.

### Trạng thái training thật — cả hai task đều CHƯA biết đi

Episode tối đa = `episode_length_s 10.0` ÷ `dt 0.005 × decimation 10` = **200 bước**.
Robot bị kết thúc sớm khi đầu thấp hơn 0.1 m hoặc nghiêng quá ngưỡng.

| Run | Task | iter | ep_len | % của 200 |
|---|---|---|---|---|
| `2026-07-23_15-23-03` | 10DOF full | 1500 | 83.1 | **41.5%** |
| `2026-07-23_15-12-33` | 10DOF6 | 350 | 32.9 | 16.5% |

Nghĩa là run tốt nhất hiện nay robot **ngã ở giây thứ 4.2 trên 10 giây**. Chưa
phải chính sách đi hoàn chỉnh, chỉ là train chưa xong (config đặt `max_iterations
= 5000`, mới chạy 1500).

Trên asset thật, **10DOF full đang nhỉnh hơn 10DOF6** — ngược với ghi chú cũ trong
`tasks/direct/transformer_nam/__init__.py` (đã đính chính trong code). Ghi chú đó
so 10DOF6 với bản 6DOF chạy trên `NewSimple.usd`, một asset đơn giản hơn hẳn.

Bốn run 10DOF, mới nhất trước:

| Run | Checkpoint | Ghi chú |
|---|---|---|
| `2026-07-27_15-02-49` | 3 | dừng ở iter 99 |
| `2026-07-23_15-23-03` | **31** | **tốt nhất — tới `model_1499`** |
| `2026-07-23_14-43-46` | 11 | tới iter 499 |
| `2026-07-23_14-33-32` | 3 | tới iter 99 |

---

## 5. Tối ưu CPU — cần chạy lại một lần

```bash
sudo bash ~/isaac-setup.sh
```

Gõ mật khẩu đăng nhập (**không hiện ký tự nào** là bình thường), Enter. Script
idempotent, chạy lại bao nhiêu lần cũng an toàn. Nó nâng swapfile 8G→16G,
zram 8G→12G, chỉnh sysctl, và cài 2 lệnh `isaac-perf` / `isaac-run`.

Sau khi cài, `run.sh` **tự động** phát hiện và dùng `isaac-run`, không cần đổi lệnh.

Lệnh dùng hằng ngày:

```bash
isaac-perf status     # xem CPU / RAM / swap / zram / trạng thái daemon
isaac-perf on         # bật hiệu năng cao thủ công
isaac-perf off        # trả về tiết kiệm điện
```

### Vì sao phải chạy lại (sửa 2026-08-03)

Bản `isaac-perf` đầu tiên có hai lỗi làm **tụt hiệu năng máy một cách âm thầm**:

1. **Ghi thẳng `/sys/firmware/acpi/platform_profile`** trong khi máy đang chạy
   `power-profiles-daemon` — daemon đó cũng quản lý đúng file này nên hai bên
   giẫm chân nhau, giá trị bị ghi đè lại.
2. **Trạng thái gốc lưu ở `/run/isaac-perf.state`**, mà `/run` bị xoá sạch mỗi lần
   khởi động máy. Mất state thì `isaac-perf off` rơi vào giá trị mặc định
   hard-code (`balanced` / `balance_performance`) — **sai với máy này**, vốn là
   `performance` / `performance`. Kết quả: cứ vài chu kỳ bật/tắt là máy tụt xuống
   `balanced` mà không ai biết.

Bản mới: state chuyển sang `/var/lib/isaac-perf.state` (sống qua reboot), bỏ sạch
giá trị đoán sẵn, và đi qua `powerprofilesctl` khi daemon đang chạy.

Kiểm tra bất cứ lúc nào bằng `powerprofilesctl get` — ra `performance` là đúng.
Nếu ra `balanced` thì `powerprofilesctl set performance` (lệnh này **không** cần sudo).

---

## 6. `num_envs` — số đo thật trên máy này (2026-08-03)

Toàn bộ số dưới đây đo bằng train thật 5 vòng, headless, task
`Transformer-Walk10DOF-Direct-v0`. `steps/s = num_envs × 24 / thời-gian-mỗi-vòng`.

### Trước và sau khi cắt buffer PhysX

| num_envs | VRAM mặc định | VRAM đã cắt | steps/s |
|---|---|---|---|
| 512 *(số cũ)* | 2437 MiB | — | 14 288 |
| 1024 | 2633 MiB | — | 24 824 |
| 2048 | 2953 MiB | — | 36 141 |
| 4096 | 3547 MiB | **1897 MiB** | 41 304 |
| 8192 | 4803 MiB | **3153 MiB** | 45 406 |
| 12288 | — | 4367 MiB | 45 511 *(không tăng nữa)* |

Cắt buffer tiết kiệm **1650 MiB ở mọi mức** — vì đó là cấp phát cố định, không phụ
thuộc `num_envs`.

### Chọn 4096, không chọn 8192

| num_envs | VRAM | RAM thấp nhất khi chạy | steps/s |
|---|---|---|---|
| **4096 (mặc định)** | 1897 MiB | **6604 Mi** | 41 304 |
| 8192 | 3153 MiB | 4341 Mi | 45 406 (+10%) |

8192 nhanh hơn 10% nhưng ăn mất 2263 Mi RAM dự phòng. Mở Firefox (~4GB) là 8192 hết
chỗ, còn 4096 vẫn dư ~2.6Gi. Muốn thêm 10% thì đóng hết ứng dụng rồi
`--num_envs 8192`. Trên 8192 **không nhanh thêm chút nào**, đừng nâng nữa.

Task 6DOF cho số gần trùng khít (4096: 1891 MiB, 6625 Mi, 40 960 steps/s).

### Buffer PhysX đã cắt — và vì sao

Đặt trong `SimulationCfg(physics=PhysxCfg(...))` của cả hai file env:

| Tham số | Mặc định | Đặt lại | Lý do |
|---|---|---|---|
| `gpu_max_soft_body_contacts` | 2²⁰ | 2¹⁰ | scene không có soft body nào |
| `gpu_max_particle_contacts` | 2²⁰ | 2¹⁰ | scene không có particle nào |
| `gpu_max_rigid_contact_count` | 2²³ | 2²⁰ | chỉ 2 bàn chân chạm đất mỗi env |
| `gpu_collision_stack_size` | 2²⁶ | 2²⁴ | nt |
| `gpu_temp_buffer_capacity` | 2²⁴ | 2²³ | nt |
| `gpu_found_lost_aggregate_pairs_capacity` | 2²⁵ | 2²² | chỉ có robot + mặt đất |

Các tham số **không** đụng tới (`gpu_found_lost_pairs_capacity`,
`gpu_total_aggregate_pairs_capacity`, `gpu_max_rigid_patch_count`,
`gpu_heap_capacity`) vì chúng co giãn theo `num_envs`, cắt vào là rủi ro.

Nếu sau này thêm vật thể vào scene mà PhysX báo overflow: nó **tự in con số nó cần**,
nâng đúng tham số đó lên nấc 2ⁿ kế tiếp, đừng nâng hết cả bảng.

---

## 7. Chống lag / tràn RAM

Máy 16GB RAM, GPU 6GB VRAM. Theo thứ tự hiệu quả:

1. **Luôn train với `--headless`.**
2. **Đóng Firefox nếu muốn dùng `--num_envs 8192`.** Ở mặc định 4096 thì không cần.
3. Khi play để `--num_envs 1`.

Hai điều **không** giúp gì, đã đo:

- **Swap không nâng được `num_envs`.** Trần nằm ở bộ nhớ PhysX cấp phát sẵn, không
  phải swap. Nếu training phải swap thật thì tốc độ sập chứ không chạy thêm được env.
  Hiện 28Gi swap, dùng **0B**.
- **Đóng ứng dụng không giải phóng VRAM.** Máy dùng đồ hoạ lai — desktop chạy trên
  iGPU Intel, RTX 4050 gần như trống sẵn (Xorg chỉ 4 MiB). Đóng app giúp RAM, không
  giúp VRAM.

Lỗi "tràn RAM" trước đây gần như chắc chắn là do buffer PhysX mặc định quá lớn cộng
với `num_envs` cao — không phải thiếu swap.

### Play có cửa sổ — số đo thật (2026-08-03)

Từ lúc gõ lệnh tới lúc robot bắt đầu bước:

| Lần chạy | Thời gian |
|---|---|
| **Đầu tiên** (cache nguội, sau khi khởi động máy) | **124 giây** |
| **Những lần sau** (cache đã ấm) | **23 giây** — nhanh gấp **5.4 lần** |

VRAM lúc đang chạy (`--num_envs 1`, có GUI): **~2600 MiB** / 6144. GPU ~37%. Thoải mái.

**Hai phút của lần đầu là bình thường, đừng tắt giữa chừng.** Đây là chỗ mất thời gian
nhiều nhất trong cả buổi làm việc hôm 03-08: hai lần đầu bị tắt ở giây **74** và **80**
vì tưởng máy treo — hụt mất khoảng 45 giây trước khi nó lên hình. Không có gì hỏng cả.

Trong hai phút đó Kit nạp 243 extension rồi khởi tạo renderer RTX. Nó **không** in gì
ra terminal ở đoạn giữa — nhìn từ ngoài không phân biệt được "đang chạy" với "treo".
Cách duy nhất để biết là chờ. Chờ trọn một lần rồi thì lần sau chỉ còn 23 giây.

Biết là đã xong khi terminal in dòng đầu tiên:

```
[    0] roll=+0.156 pitch=-0.226 gz=+0.008 h=0.370m | ...
```

Nếu Ubuntu hiện *"Isaac Lab is not responding"* lặp đi lặp lại, xem mục 9.

---

## 8. Những gì đã sửa trong code ngày 2026-08-03

Sao lưu toàn bộ bản trước khi sửa ở
`transformer_nam/.backup_truoc_khi_sua_20260803_010207/`.

| # | Lỗi | Sửa |
|---|---|---|
| 1 | `ImportError: cannot import name 'as_torch'` — 7 file env import hàm không tồn tại | khôi phục `as_torch` vào `_lab3_compat.py` |
| 2 | ProxyArray của IsaacLab 3.0 lọt vào 9 hàm `@torch.jit.script` | bọc `as_torch(...)` ở 56 chỗ đọc `robot.data` |
| 3 | `KeyError: 'actor_state_dict'` khi nạp checkpoint cũ | thêm `scripts/convert_checkpoint_rslrl5.py` |
| 4 | `'PPO' object has no attribute 'actor_critic'` | `play.py` dùng `alg.get_policy()` của rsl-rl ≥ 4.0 |
| 5 | `IndexError: index 60 out of bounds` | `play.py` hết hard-code cho task TwistMarch 62D |
| 6 | `run.sh` / `run_direct.sh` / `isaaclab.sh` trỏ đường dẫn chết, sai tên conda env | viết lại, gom về `run.sh` |

Lỗi 1–2 do sửa nhầm từ trước. Lỗi 3–5 là hệ quả tất yếu khi project chuyển từ
IsaacLab 2.x lên 3.0.

### Đợt sau, cùng ngày — dọn hạ tầng cho khớp laptop

| # | Việc | Kết quả |
|---|---|---|
| 7 | `isaac-perf` đánh nhau với `power-profiles-daemon`, state ở `/run` mất sau reboot | viết lại (mục 5) — **cần chạy lại `sudo bash ~/isaac-setup.sh`** |
| 8 | `import transformer_nam` gãy ngoài SimulationApp vì `from .ui_extension_example import *` | bỏ dòng import (boilerplate template, project không dùng) |
| 9 | Buffer PhysX để mặc định, sized cho manipulation | thêm `PhysxCfg` vào cả 2 env — tiết kiệm **1650 MiB** |
| 10 | `num_envs=512` kế thừa từ workstation | đo lại, đổi thành **4096** — nhanh gấp **2.9 lần** |
| 11 | Chưa biết checkpoint cũ còn giá trị sau migration | thí nghiệm đối chứng — **còn nguyên giá trị** (mục 4) |
| 12 | Ghi chú convergence trong `tasks/.../__init__.py` so sánh lệch chuẩn | đính chính bằng số đo thật |
| 13 | 81 run không rõ cái nào dùng task nào | thêm `logs/README-runs.md` |
| 14 | `outputs/` 15M rác hydra, run rỗng, file 0 byte | xoá |

### Đợt thứ ba, 2026-08-03 (đêm sang 04) — mở được cửa sổ Isaac Sim khi play

Triệu chứng ban đầu: `play.py` chạy ngon, in đủ log, robot cử động — nhưng **không có
cửa sổ nào**. Kết quả cuối: play mở cửa sổ, đã xem được cả policy tháng 7 lẫn policy
tháng 2.

#### Lỗi có sẵn trong project

| # | Lỗi | Vì sao xảy ra | Cách sửa |
|---|---|---|---|
| 15 | Không mở được cửa sổ dù bỏ `--headless` | IsaacLab 3.0 **đảo ngược mặc định**: không truyền `--viz` là headless. `--headless` giờ chỉ là cờ deprecated, vẫn nhận nhưng hết tác dụng | `play.py` tự gán `--viz kit` khi người dùng không nói gì |
| 16 | Play tua quá nhanh, nhìn không kịp | không ai đặt `--real-time` | cùng nhánh trên bật luôn `real_time` |
| 17 | Render có thể rơi xuống iGPU Intel | máy đồ hoạ lai, `prime-select` ở chế độ `on-demand`, desktop chạy trên Intel | thêm 3 biến `__NV_PRIME_RENDER_OFFLOAD` / `__VK_LAYER_NV_optimus` / `__GLX_VENDOR_LIBRARY_NAME` vào `run.sh` |
| 18 | Hộp thoại *"not responding"* hiện lại chục lần | GNOME `check-alive-timeout` = **5 giây**, mà Kit đứng lâu hơn thế liên tục suốt 2 phút khởi động | đặt về 0 — xem mục 9 |
| 19 | Cứ tưởng máy treo nên tắt giữa chừng | không ai biết mốc bình thường là bao nhiêu | đo: **124 giây** lần đầu, **23 giây** lần sau — mục 7 |
| 20 | Log in `TwL`/`TwR` nhưng đọc **nhầm khớp** | chú thích trong `play.py` ghi thứ tự khớp là "trái hết rồi tới phải", còn USD lại **xen kẽ** trái/phải. Nên index 2 là `Hipleft`, index 7 là `Kneeright` — không phải hai khớp xoay (thật ra ở **4** và **5**) | tra chỉ số **theo tên khớp** qua `robot.joint_names`, và in ra lúc khởi động để đối chiếu |

#### Hai lỗi do CHÍNH đợt sửa này gây ra

Ghi lại đầy đủ vì cả hai đều là kiểu sai dễ lặp lại.

| Lỗi tự gây | Lý do đã làm sai | Bắt được nhờ | Cách sửa |
|---|---|---|---|
| `--viz none` (cách chính thức để ép headless) bị ghi đè thành **mở cửa sổ** | Nhánh tự bật GUI kiểm tra `if args_cli.visualizer is None`. Nhưng `AppLauncher._parse_visualizer_csv` trả về `None` cho **cả hai** ca: "không truyền gì" **và** "truyền `none`". Nên đã hiểu nhầm "người dùng ép tắt" thành "người dùng không nói gì" | Chạy thử 5 tổ hợp cờ trước khi tin là xong | Đổi sang cờ `visualizer_explicit` mà `ExplicitAction` đặt sẵn — cờ này chỉ bật khi cờ được gõ ra thật, phân biệt được hai ca |
| `--rendering_mode performance` làm **viewport đen hoàn toàn** | Thêm vào với lý do "GPU chỉ 6GB nên giảm tải cho nhẹ", tự đánh giá là an toàn và chắc chắn có lợi, rồi **không kiểm bằng mắt**. Thực tế preset đó tắt quá nhiều tính năng RTX nên không dựng được gì — chứ không phải "dựng xấu hơn cho nhanh" | Bạn chỉ ra ảnh viewport đen so với GIF mẫu. Sau đó chụp màn hình đối chứng: cùng checkpoint, có cờ → đen, bỏ cờ → hiện robot | Gỡ hẳn, để mặc định Kit. Xem mục 10 |

> **Vì sao lỗi thứ hai lọt lâu:** sim vẫn chạy, log vẫn in đẹp đều đặn. Nhìn từ terminal
> **không thể biết viewport đang đen**. Đã kiểm chứng bằng log rồi tưởng là xong.
>
> Rút ra: **thay đổi gì về đồ hoạ thì phải kiểm bằng mắt, không kiểm bằng log.** Và
> "tối ưu cho máy yếu" nghe hợp lý nên rất dễ thêm vào mà quên kiểm chứng — preset render
> không phải cái núm xoay tuyến tính, tắt sai thứ là mất hình hẳn.

#### Ba bài học chung

> **1. Nâng cấp framework lớn thì thứ vỡ đau nhất là mặc định bị đảo ngược.** Cờ cũ vẫn
> còn, vẫn nhận, không báo lỗi — chỉ là hết tác dụng. Tệ hơn API gãy, vì API gãy thì nổ
> ngay còn cái này im lặng làm sai. `guidance.md` từng ghi "bỏ `--headless` để hiện cửa
> sổ" — đúng với 2.x, sai với 3.0, và làm mất gần một buổi.
>
> **2. "Máy treo" và "máy đang chạy nhưng không nói gì" nhìn giống hệt nhau.** Đã tắt
> nhầm hai lần ở giây 74 và 80 trong khi đích là giây 124. Trước khi kết luận treo, phải
> biết mốc bình thường là bao nhiêu.
>
> **3. Chỉ số vào mảng thì tra theo tên, đừng viết cứng.** Thứ tự khớp do file USD quyết
> định, không phải do code — đổi asset là lệch hết mà không báo lỗi. Manh mối để lộ ra
> vụ `TwL`/`TwR` là **giới hạn khớp**: số báo ±25° trong khi khớp xoay cho phép ±180°,
> còn khớp hông thì đúng ±37°. Số không bao giờ chạm gần giới hạn của khớp mà nó tự nhận
> là dấu hiệu đang đọc nhầm khớp.

---

## 9. Hộp thoại "Isaac Lab is not responding" hiện lặp đi lặp lại

Bấm "Wait" bao nhiêu lần nó cũng hiện lại. **Không phải Isaac Sim hỏng.**

GNOME ping cửa sổ theo chu kỳ; quá `check-alive-timeout` mà không trả lời thì báo treo.
Mặc định là **5000 ms**, trong khi Kit đứng lâu hơn thế liên tục suốt hai phút khởi động —
nên hộp thoại bật đi bật lại.

```bash
gsettings set org.gnome.mutter check-alive-timeout 0    # 0 = tắt hẳn kiểm tra treo
```

Hoàn tác:

```bash
gsettings reset org.gnome.mutter check-alive-timeout
```

Đây là thiết lập **desktop**, không phải của project — nó áp cho mọi ứng dụng, và có
nghĩa là app treo thật cũng sẽ không được GNOME báo nữa. Muốn giữ cảnh báo cho app khác
thì đặt một giá trị lớn thay vì 0, ví dụ `60000` (60 giây).

---

## 10. Cửa sổ mở ra nhưng viewport ĐEN — đừng đụng `--rendering_mode`

Triệu chứng: cửa sổ Isaac Lab hiện bình thường, cây Stage có đủ `World`/`physicsScene`,
terminal in log robot chạy đều — nhưng khung 3D **đen thui**, không thấy cả lưới sàn.

Nguyên nhân: chạy kèm `--rendering_mode performance`. Preset đó
(`~/IsaacLab/apps/rendering_modes/performance.kit`) tắt hàng loạt tính năng RTX —
ambient occlusion, reflections, indirect diffuse, hạ `maxBounces` xuống 2. Trên máy này
kết quả là **không dựng được gì cả**, chứ không phải "dựng xấu hơn cho nhanh".

Cách sửa: **đừng truyền cờ đó.** Mặc định của Kit hiện đủ và vẫn mượt — `--num_envs 1`
chỉ dùng ~2600 MiB VRAM, GPU ~37%, thừa sức.

Đã đo bằng chụp màn hình đối chứng: cùng một checkpoint, `performance` cho viewport đen,
bỏ cờ đi thì robot hiện rõ trên lưới sàn.

> **Bài học:** "tối ưu cho máy yếu" nghe hợp lý nên rất dễ thêm vào mà không kiểm chứng.
> Nhưng preset render không phải cái núm xoay tuyến tính — tắt sai thứ là mất hình hẳn.
> Tệ hơn nữa, sim vẫn chạy và log vẫn in đẹp, nên nhìn từ terminal **không thể biết
> viewport đang đen**. Bất cứ thay đổi nào về đồ hoạ đều phải kiểm bằng mắt, không kiểm
> bằng log.

---

### Hai cảnh báo `pip check` — cố ý KHÔNG sửa

```
isaaclab-rl 0.5.5   yêu cầu packaging<24   (đang có 26.0)
isaacsim-kernel     yêu cầu coverage==7.4.4 (đang có 7.6.1)
```

Cả hai là pin cũ trong metadata, vô hại. `train.py` chỉ dùng `packaging` cho
`version.parse` — đã chạy thật nhiều lần không vấn đề gì; `coverage` chỉ dùng khi
chạy test. Hạ cấp hai gói này rủi ro hơn nhiều so với để nguyên.

### Đã kiểm chứng bằng chạy thật

- Train 2 task × 2 mức `num_envs`, **0 lỗi, 0 cảnh báo PhysX overflow**.
- Resume từ `model_1499_rslrl5.pt`: `iter` khôi phục đúng, ep_len leo 23.9 → 69.7.
- Train từ đầu làm đối chứng: ep_len đứng yên 18.5 → 18.0.
- Play 10DOF 120 giây, 2359 dòng log, robot chuyển động bình thường.

Đợt thứ ba (mở cửa sổ), kiểm bằng **cả log lẫn ảnh chụp màn hình**:

| Kiểm tra | Kết quả |
|---|---|
| Play 10DOF có cửa sổ, policy tháng 7 `model_1499` | robot hiện trên lưới sàn, chạy tới step **4246** |
| Play 6DOF có cửa sổ, policy tháng 2 `model_999` | nạp được, chạy tới step **8618**, 0 lỗi |
| Chỉ số khớp sau khi sửa | in ra `Twistleft=[4] Twistright=[5]` — khớp với bảng Isaac Sim |
| 5 tổ hợp cờ (trần / `--headless` / `--viz none` / `--viz kit` / `--viz rerun`) | **5/5 đạt** sau khi sửa lỗi `--viz none` |
| `--headless` vẫn chạy ngầm | 0 visualizer, 2217 dòng log |
| Train không bị ảnh hưởng | `--max_iterations 3` exit 0, ep_len 15–18 (đúng mốc "train từ đầu") |
| VRAM khi play có GUI | ~2600 MiB / 6144, GPU ~37% |
