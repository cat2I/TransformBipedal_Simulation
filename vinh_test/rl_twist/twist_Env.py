import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class twistEnv(gym.Env):
    def __init__(self, xml_file="Fulltrans_RL.xml"):
        super().__init__()

        # nạp thế giới vật lí
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # tần số ra quyết định của policy: 10hz
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip

        # action space:  10 động cơ
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),  # self.model.nu = Số lượng động cơ = 10
            dtype=np.float32,
        )
        # observation space: 2 góc nghiêng pitch/roll, 10 position + 10 vel = 22
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

    # hàm thu thập dữ liệu state / observations
    def _get_obs(self):
        # 1. Lấy góc nghiêng (Orientation) của thân từ MuJoCo (định dạng Quaternion [w, x, y, z])
        quat = self.data.qpos[3:7]
        w, x, y, z = quat

        # Công thức toán học chuyển Quaternionsang góc Roll (Nghiêng trái/phải) và Pitch (Ngảtrước/sau)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # 2. Lấy góc thực tế của 10 động cơ
        joint_positions = self.data.qpos[7:]

        # 3. Lấy vận tốc quay của 10 động cơ
        joint_velocities = self.data.qvel[6:]

        # 4. Gộp (concatenate) tất cả lại thành 1 mảng 1D duy nhất
        obs = np.concatenate(
            [[roll, pitch], joint_positions, joint_velocities]  # 2 số  # 10 số  # 10 số
        ).astype(np.float32)

        return obs

    # hàm reset về vị trí ban đầu
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset lực và vận tốc cũ
        mujoco.mj_resetData(self.model, self.data)

        # đặt base cao z = 0.42
        self.data.qpos[2] = 0.42

        # đặt initial pose — GHI THEO TÊN KHỚP, không theo thứ tự mảng.
        # MuJoCo xếp khớp theo TỪNG CHÂN (Bubleft,Hipleft,Twistleft,Kneeleft,
        # Footleft, rồi mới sang chân phải), KHÔNG phải cặp trái-phải như Isaac.
        # Ghi bằng array theo vị trí sẽ đặt sai 8/10 khớp. Dùng dict theo tên thì
        # thứ tự khớp trong XML không còn quan trọng — luôn đặt đúng khớp.
        nominal_pose = {
            "Bubleft_joint": 0.07,
            "Bubright_joint": 0.07,
            "Hipleft_joint": 0.0,
            "Hipright_joint": 0.0,
            "Twistleft_joint": 1.51,  # chân đang chĩa ra 2 bên (tư thế xoạc)
            "Twistright_joint": 1.51,
            "Kneeleft_joint": -0.044,
            "Kneeright_joint": -0.044,
            "Footleft_joint": -0.107,
            "Footright_joint": -0.107,
        }

        # cộng nhiễu ngẫu nhiên rồi ghi vào đúng địa chỉ qpos của từng khớp theo tên
        for name, angle in nominal_pose.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = angle + np.random.uniform(-0.05, 0.05)

        # Yêu cầu MuJoCo cập nhật lại hình dáng ật lý
        mujoco.mj_forward(self.model, self.data)

        # Trả về góc nhìn (Mở mắt ra đầu game)
        observation = self._get_obs()
        info = {}
        return observation, info

    # hàm step
    def step(self, action):
        # giới hạn mạng neural chỉ dc vặn 0.1 rad mỗi step
        action_scale = 0.1

        current_joint_pos = self.data.qpos[7:]  # lay 10 goc hien tai

        # góc mục tiêu = góc hiện tai + action *0,1
        target_pos = current_joint_pos + (action * action_scale)

        # gửi lệnh xuống cho 10 động cơ của mjc
        self.data.ctrl[:] = target_pos

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 3. MỞ MẮT RA NHÌN SAU KHI NHÚC NHÍCH
        observation = self._get_obs()

        # 4. TÍNH ĐIỂM THƯỞNG

        # a. Lấy dữ liệu cần thiết
        base_z = self.data.qpos[2]  # Chiều cao ủa hông so với mặt đất
        roll, pitch = observation[0], observation[1]  # Lấy từ mảng observation

        # Tìm địa chỉ chính xác của khớp Twist rong mảng qpos bằng tên
        twist_L_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Twistleft_joint"
        )
        twist_R_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Twistright_joint"
        )

        # Đọc góc hiện tại của khớp Twist
        twist_left = self.data.qpos[self.model.jnt_qposadr[twist_L_id]]
        twist_right = self.data.qpos[self.model.jnt_qposadr[twist_R_id]]

        # REWARD
        survival_reward = 1.0  # thưởng sinh tồn
        # error càng cần 0 thì điểm thưởng càng nhiều theo hàm số e mũ. max là 2.0
        twist_error = abs(twist_left) + abs(twist_right)
        twist_reward = 2.0 * np.exp(-3.0 * twist_error)

        # phạt năng lượng: trừ điểm nếu xuất action quá mạnh
        energy_penalty = 0.02 * np.sum(np.square(action))

        # tổng điểm
        reward = survival_reward + twist_reward - energy_penalty

        # kiểm tra xem có bị ngã không:
        terminated = False

        # điều kiện ngã: base thấp hoặc nghiêng quá 30 độ

        if base_z < 0.32 or abs(roll) > 0.5 or abs(pitch) > 0.5:
            terminated = True
            reward -= 100.0

        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
