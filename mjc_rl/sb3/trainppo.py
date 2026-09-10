from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from twist_Env import twistEnv

# Trong so va tensorboard ghi vao mjc_rl/model/, khong ghi vao thu muc dang dung.
# Tinh tu vi tri file nay nen chay tu dau cung ra dung cho.
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def main():
    print("1. Đang tải môi trường vật lý MuJoCo...")
    # Tạo môi trường mà ta vừa tự code lúc nãy
    env = twistEnv(xml_file="Fulltrans_RL.xml")

    print("2. Đang kiểm tra lỗi môi trường...")
    # Công cụ này của SB3 sẽ giả lập chơi thử vàivòng để bắt lỗi code (nếu có)
    check_env(env)
    print("Môi trường đạt chuẩn Gymnasium! Không cólỗi.")

    print("3. Bắt đầu cấy não PPO...")
    # Khởi tạo mô hình mạng nơ-ron
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log=str(MODEL_DIR / "ppo_twist_tensorboard"))

    print("4. Bấm giờ đi học (Training) - Có thể mất từ vài phút đến vài chục phút...")
    # total_timesteps là tổng số bước (step) robot sẽ thử nghiệm.
    # Mới test thử, ta để 200,000 bước. (Thực tế khi train thật có thể cần 2-3 triệu bước).
    model.learn(total_timesteps=200000)

    print("5. Học xong, đang xuất chuồng!")
    # Lưu toàn bộ "chất xám" (Trọng số mạng nơ-ron) vào file nén .zip
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR / "ppo_twist_model"))
    print(f"Đã lưu trọng số vào {MODEL_DIR / 'ppo_twist_model.zip'} thành công!")


if __name__ == "__main__":
    main()
