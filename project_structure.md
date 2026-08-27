# Tài liệu Cấu trúc Dự án: Transform_bipedal

Dự án **Transform_bipedal** là một dự án ứng dụng Học tăng cường (Reinforcement Learning - RL) trong môi trường mô phỏng **Isaac Sim** (thông qua **IsaacLab** và **RSL_RL**) để huấn luyện một robot hai chân (bipedal robot) thực hiện các hành động như đi lại, đứng dậy.

Dưới đây là tài liệu phân tích sâu (deep dive) về cấu trúc thư mục, cũng như mục đích cụ thể của từng file mã nguồn trong dự án.

---

## 1. Cấu trúc Thư mục Tổng quan

```text
Transform_bipedal/
├── transformer_nam/      # Thư mục cốt lõi chứa toàn bộ mã nguồn RL và môi trường Isaac Sim.
├── urdf/                 # Chứa các file mô hình URDF ban đầu của robot.
├── FullForm/             # Chứa mô hình robot dạng đầy đủ (USD & config).
├── 3DOFTrans/            # Chứa mô hình robot dạng 3 bậc tự do (3-DOF) mỗi chân.
├── NewSimple/            # Chứa mô hình robot dạng đơn giản hóa.
├── docs/                 # Tài liệu về cấu hình RL.
├── README.md             # Hướng dẫn tổng quan và cách chạy script cơ bản.
├── CAI-DAT-ISAACLAB.md   # Hướng dẫn cài đặt IsaacLab cho dự án.
├── guidance.md           # Hướng dẫn chi tiết / cẩm nang cho dự án.
└── mujoco_view.py        # Script phụ trợ để xem mô hình URDF/XML bằng thư viện MuJoCo.
```

---

## 2. Phân tích Sâu Thư mục Cốt lõi: `transformer_nam/`

Đây là nơi chứa toàn bộ logic về môi trường mô phỏng, phần thưởng (reward function), cấu hình huấn luyện (training config) và các tập lệnh (scripts) để chạy mô hình.

### 2.1. Các file Shell Scripts (Tập lệnh chạy)
Các file này dùng để thiết lập biến môi trường và chạy dự án một cách thuận tiện nhất (tự động kích hoạt conda, setup PYTHONPATH).

- **`run.sh` / `run_direct.sh`**: Entry-point chính để chạy huấn luyện hoặc test. Nó xử lý việc kích hoạt môi trường conda `isaacsim` và gọi các script Python bên trong.
- **`isaaclab.sh`**: Script tiện ích để chạy các tác vụ liên quan trực tiếp đến gốc của IsaacLab.

### 2.2. Thư mục `scripts/`
Chứa các script Python thực thi trực tiếp, phục vụ cho việc huấn luyện, kiểm tra và debug.

- **`rsl_rl/train.py`**: File chính để **khởi động quá trình huấn luyện** (training) sử dụng thư viện RSL_RL. Nó đọc cấu hình môi trường, cấu hình thuật toán PPO và bắt đầu vòng lặp RL.
- **`rsl_rl/play.py`**: File dùng để **chạy thử (inference)** một policy đã được huấn luyện (load từ file `.pt`).
- **`rsl_rl/cli_args.py`**: Xử lý các tham số dòng lệnh (command-line arguments) cho quá trình train/play (như `--num_envs`, `--task`, `--headless`, v.v.).
- **`rsl_rl/export_trajectory.py`**: Script để xuất quỹ đạo (trajectory) của robot sau khi policy sinh ra hành động, có thể dùng để phân tích offline.
- **`debug_robot_position.py` / `inspect_usd_structure.py`**: Các công cụ để debug trực quan, kiểm tra tọa độ và cấu trúc cây (tree structure) của file USD trong Isaac Sim để đảm bảo import model đúng.
- **`check_joint_order.py`**: Kiểm tra thứ tự các khớp (joints) của robot để đảm bảo action do mạng Neural Network xuất ra khớp với đúng động cơ trên Isaac Sim.
- **`convert_checkpoint_rslrl5.py`**: Script hỗ trợ chuyển đổi/tương thích các file checkpoint mô hình giữa các phiên bản RSL_RL.
- **`list_envs.py`**: Liệt kê các môi trường đã được đăng ký (registered) trong project.
- **`play_transform_getup.py` / `random_agent.py` / `zero_agent.py`**: Các script chạy thử robot với các hành động ngẫu nhiên, hoặc lực bằng 0 (để test vật lý rơi tự do), hoặc test chuyên biệt cho tác vụ đứng dậy (getup).

### 2.3. Thư mục `source/transformer_nam/transformer_nam/tasks/`
Nơi định nghĩa **Môi trường RL (RL Environments)**. Dự án được chia làm 2 phương pháp tiếp cận môi trường (theo tiêu chuẩn của IsaacLab): `direct` (trực tiếp) và `manager_based` (dựa trên trình quản lý).

#### Thư mục `direct/` (Direct Workflow)
Cách tiếp cận này viết môi trường trực tiếp, hiệu năng cao, can thiệp sâu vào các tensor vật lý. Đa số code tập trung ở đây.

- **`transformer_nam_env.py` / `transformer_nam_env_cfg.py`**: Định nghĩa lớp môi trường chính (`TransformerWalkEnv`) và file cấu hình đi kèm. Chứa các logic quan trọng như:
  - Khởi tạo vật lý.
  - Tính toán bước vật lý (step).
  - Các hàm **Reward (Phần thưởng)**: `orientation_reward`, `deviation_reward`, `height_reward`, `velocity_reward`, v.v. để định hướng cho robot đi bộ.
- **`transformer_walk10dof_env.py` / `transformer_walk10dof6_env.py`**: Các biến thể môi trường dành riêng cho cấu hình robot có 10 bậc tự do (10-DOF). Mỗi file có thể tinh chỉnh hàm reward hoặc không gian quan sát (observation space) một chút để tối ưu.
- **`transformer_nam_env_3dof.py` / `transformer_nam_env_4dof.py`**: Biến thể môi trường cho các mô hình robot đơn giản hóa có 3 hoặc 4 bậc tự do mỗi chân.
- **`transformer_hieu_env.py` / `transform_hieu_env_work1.py`**: Các biến thể thử nghiệm do tác giả "Hiếu" phát triển, tập trung vào các tác vụ như đứng tại chỗ (Stand) hoặc đi bộ xoắn (TwistMarch).
- **`transformer_config.py`** (và các biến thể `_3dof`, `_4dof`, `_10dof`): Quản lý các cấu hình chi tiết (scale của action, hệ số PD control, khoảng giới hạn của khớp) áp dụng cho các phiên bản môi trường tương ứng.
- **`agents/rsl_rl_ppo_cfg.py`**: Cấu hình thuật toán PPO của RSL_RL. Chứa các siêu tham số (hyperparameters) như learning rate, batch size, số vòng lặp PPO, kiến trúc mạng Neural Network (ví dụ: `[512, 256, 128]`).

#### Thư mục `manager_based/` (Manager-Based Workflow)
Cách tiếp cận này sử dụng các Manager của IsaacLab (SceneManager, ActionManager, RewardManager...) giúp module hóa code tốt hơn.
- **`transformer_nam_env_cfg.py`**: File cấu hình tổng hợp định nghĩa toàn bộ Scene, Observation, Action và Event.
- **`mdp/rewards.py`**: Các hàm tính toán phần thưởng (Reward functions) được tách rời và gọi qua hàm callback.
- **`agents/rsl_rl_ppo_cfg.py`**: Cấu hình PPO tương tự như phần Direct, nhưng áp dụng cho môi trường Manager-based.

### 2.4. Các file cấu hình Python Package (Boilerplate)
- **`pyproject.toml` / `setup.py` / `config/extension.toml`**: Các file tiêu chuẩn để đóng gói thư mục `transformer_nam` thành một Python module và một Isaac Sim Extension, giúp hệ thống có thể import `transformer_nam` từ bất kỳ đâu.

---

## 3. Thư mục Mô hình Vật lý (URDF / USD)

Robot được thiết kế ở nhiều phiên bản khác nhau, được lưu trữ trong các thư mục riêng biệt:

- **`urdf/`**: Chứa các file `.urdf` và `.csv` gốc. URDF là định dạng chuẩn của ROS mô tả các link (khối), joint (khớp), khối lượng và quán tính của robot.
- **`FullForm/`**: Mô hình robot đầy đủ nhất.
  - `Fulltrans10DOF.usd`: File USD (Universal Scene Description - định dạng chuẩn của NVIDIA Omniverse) đã được chuyển đổi từ URDF để tối ưu cho Isaac Sim.
  - `config/joint_names_FullForm.yaml`: Mapping tên các khớp của robot.
- **`3DOFTrans/` & `NewSimple/`**: Các mô hình rút gọn, phục vụ cho việc huấn luyện các phiên bản kiểm chứng (proof of concept) để giải quyết bài toán dễ trước khi áp dụng vào con robot đầy đủ phức tạp.

---

## Tổng kết Lồng ghép Workflow
1. **Khởi tạo**: Robot được thiết kế và xuất ra URDF.
2. **Chuẩn bị**: URDF được nạp vào Isaac Sim để chuyển đổi thành USD (`FullForm/Fulltrans10DOF.usd`).
3. **Môi trường**: Lớp `TransformerWalkEnv` (`tasks/direct/...`) tải file USD này lên GPU, tạo ra hàng ngàn bản sao (vectorized environments).
4. **Huấn luyện**: Chạy `./run.sh scripts/rsl_rl/train.py --task Transformer-Direct-v0`. Quá trình sẽ sử dụng cấu hình PPO (`agents/rsl_rl_ppo_cfg.py`) để tối ưu hóa hành vi đi bộ.
5. **Đánh giá**: Chạy `./run.sh scripts/rsl_rl/play.py` để xem robot thực tế hoạt động như thế nào thông qua UI của Isaac Sim.
