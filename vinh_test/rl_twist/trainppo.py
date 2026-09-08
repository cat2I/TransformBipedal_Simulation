import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from twist_Env import twistEnv


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
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_twist_tensorboard/")

    print("4. Bấm giờ đi học (Training) - Có thể mất từ vài phút đến vài chục phút...")
    # total_timesteps là tổng số bước (step) robot sẽ thử nghiệm.
    # Mới test thử, ta để 200,000 bước. (Thực tế khi train thật có thể cần 2-3 triệu bước).
    model.learn(total_timesteps=200000)

    print("5. Học xong, đang xuất chuồng!")
    # Lưu toàn bộ "chất xám" (Trọng số mạng nơ-ron) vào file nén .zip
    model.save("ppo_twist_model")
    print("Đã lưu trọng số vào file ppo_twist_model.zip thành công!")


if __name__ == "__main__":
    main()
