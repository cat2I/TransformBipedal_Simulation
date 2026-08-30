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

        # giới hạn thời gian để tránh chạy vô tận
        self.max_steps = 300
        self.current_step = 0

        # thêm ma sát xoắn -tránh xoay trực tiếp khớp twist
        for i in range(self.model.ngeom):
            if self.model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE:
                self.model.geom_condim[i] = 4
                self.model.geom_friction[i][1] = 0.05  # Lực cản khi xoay vặn
                self.model.geom_friction[i][2] = 0.005  # Lực cản khi lăn

        # ---- Lưu giá trị GỐC cho domain randomization ----------------------
        # Randomization trong reset() sẽ tính TỪ các giá trị gốc này (nominal),
        # KHÔNG nhân vào giá trị hiện tại. Nếu nhân dồn vào giá trị hiện tại thì
        # sau vài trăm tập khối lượng/ma sát sẽ trôi ra vô cực.
        self._nonplane_geoms = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
        ]
        self._base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Baselink"
        )
        self._nominal_mass = float(self.model.body_mass[self._base_id])
        self._nominal_ipos = self.model.body_ipos[self._base_id].copy()
        self._nominal_actfrc = self.model.jnt_actfrcrange.copy()

        # action space:  10 động cơ, thêm 2 tín hiệu chạm sàn 0 là
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),  # self.model.nu = Số lượng động cơ = 10
            dtype=np.float32,
        )
        # observation space: 2 góc nghiêng pitch/roll, 10 position + 10 vel = 22
        # thêm 2 tín hiệu chạm sàn contact trái phải -24 tín
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

    def get_foot_contact(self):
        contact_L, contact_R = 0.0, 0.0
        for i in range(self.data.ncon):
            geom1_body = self.model.geom_bodyid[self.data.contact[i].geom1]
            geom2_body = self.model.geom_bodyid[self.data.contact[i].geom2]

            body1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, geom1_body)
            body2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, geom2_body)

            if body1 and body2:
                if "Footleft" in body1 or "Footleft" in body2:
                    contact_L = 1.0
                if "Footright" in body1 or "Footright" in body2:
                    contact_R = 1.0
        return contact_L, contact_R

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

        # lấy contact bàn chân
        contact_L, contact_R = self.get_foot_contact()

        # 4. Gộp (concatenate) tất cả lại thành 1 mảng 1D duy nhất
        obs = np.concatenate(
            [
                [roll, pitch],
                joint_positions,
                joint_velocities,
                [contact_L, contact_R],
            ]  # 2 số  # 10 số  # 10 số + 2 contact sensor o chan
        ).astype(np.float32)
        return obs

    def _randomize(self):
        """Domain randomization: mỗi tập một bộ tham số vật lí khác nhau.

        Luôn tính TỪ giá trị gốc (nominal) đã lưu ở __init__, không nhân dồn.
        Ép policy học chiến lược bền với sai khác sim-thực (ma sát, khối lượng,
        trọng tâm, sức động cơ) thay vì overfit vào một sim cố định.
        """
        # ma sát bàn chân: trượt + xoắn — sàn thật không biết trước
        slide = np.random.uniform(0.6, 1.4)
        torsion = np.random.uniform(0.02, 0.12)
        for i in self._nonplane_geoms:
            self.model.geom_friction[i][0] = slide
            self.model.geom_friction[i][1] = torsion

        # khối lượng base: robot thật nặng hơn CAD
        self.model.body_mass[self._base_id] = self._nominal_mass * np.random.uniform(
            0.8, 1.5
        )

        # trọng tâm base: lắp pin/mạch làm lệch CoM ±2 cm mỗi trục
        self.model.body_ipos[self._base_id] = self._nominal_ipos + np.random.uniform(
            -0.02, 0.02, size=3
        )

        # trần mô-men động cơ: servo yếu đi khi nóng/quay nhanh
        self.model.jnt_actfrcrange = self._nominal_actfrc * np.random.uniform(0.7, 1.0)

    # hàm reset về vị trí ban đầu
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # đếm lại số bước cho tập mới (nếu không, số đếm trôi qua các tập)
        self.current_step = 0

        # đổi tham số vật lí cho tập này
        self._randomize()

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
        self.current_step += 1
        action_scale = 0.1

        # clamp action transh vang quas ddaf
        action = np.clip(action, -1.0, 1.0)

        current_joint_pos = self.data.qpos[7:]  # lay 10 goc hien tai

        # góc mục tiêu = góc hiện tai + action *0,1
        target_pos = current_joint_pos + (action * action_scale)

        # gửi lệnh xuống cho 10 động cơ của mjc
        self.data.ctrl[:] = target_pos

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 3. MỞ MẮT RA NHÌN SAU KHI NHÚC NHÍCH
        observation = self._get_obs()
        # lấy 2 biến contact ở cuối mảng observation
        contact_L, contact_R = observation[-2], observation[-1]

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

        # Đọc VẬN TỐC hiện tại của khớp Twist
        twist_L_vel = self.data.qvel[self.model.jnt_dofadr[twist_L_id]]
        twist_R_vel = self.data.qvel[self.model.jnt_dofadr[twist_R_id]]

        # REWARD

        # error càng cần 0 thì điểm thưởng càng nhiều theo hàm số e mũ. max là 2.0
        twist_error = abs(twist_left) + abs(twist_right)
        twist_reward = 2.0 * np.exp(-1.0 * twist_error)

        # hàm để ép điểm: đứng im thì lỗ
        progress_reward = 3.0 - twist_error

        # phạt nghiên base quá
        orientation_penalty = 2.0 * (abs(roll) + abs(pitch))

        # phạt năng lượng: trừ điểm nếu xuất action quá mạnh
        energy_penalty = 0.02 * np.sum(np.square(action))

        # phạt nhấc chân lên: nếu đang chạm đất mà cố tình twist thì trừ
        twist_penalty = 0.0
        if contact_L > 0.5:
            twist_penalty += abs(twist_L_vel) * 0.5  # vận tốc càng lớn phạt càng nhiều
        if contact_R > 0.5:
            twist_penalty += abs(twist_R_vel) * 0.5

        # tổng điểm
        reward = (
            progress_reward
            - energy_penalty
            - twist_penalty
            + twist_reward
            - orientation_penalty
        )

        # kiểm tra xem có bị ngã không:
        terminated = False

        # điều kiện ngã: base thấp hoặc nghiêng quá 30 độ

        if base_z < 0.32 or abs(roll) > 0.5 or abs(pitch) > 0.5:
            terminated = True
            reward -= 100.0

        # hết thời gian tập (không phải ngã) — cắt để tập không chạy vô tận
        truncated = self.current_step >= self.max_steps
        info = {}

        return observation, reward, terminated, truncated, info
