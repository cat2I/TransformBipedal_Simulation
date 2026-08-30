import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from twist_Env import twistEnv

def play():
    print("1. Tải môi trường và Bộ não...")
    env = twistEnv(xml_file="Fulltrans_RL.xml")
    
    # Load trọng số mạng nơ-ron từ file zip
    try:
        model = PPO.load("ppo_twist_model")
    except Exception as e:
        print("Lỗi: Không tìm thấy file ppo_twist_model.zip.")
        return
        
    # Đặt robot về tư thế xoạc chân ban đầu
    obs, info = env.reset()
    
    print("2. Mở cửa sổ đồ họa 3D. Nhấn ESC hoặc nút [X] để thoát.")
    
    # Kích hoạt cửa sổ MuJoCo 3D Viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # HỎI BỘ NÃO
            action, _states = model.predict(obs, deterministic=True) 
            
            # Thực thi hành động xuống môi trường
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Nếu robot ngã sấp mặt, hoặc bị hết giờ -> Đặt lại vị trí ban đầu
            if terminated or truncated:
                obs, info = env.reset()
                
            # Cập nhật hình ảnh đồ họa
            viewer.sync()
            
            # Neo tốc độ khung hình (60Hz)
            time_until_next_step = env.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    play()
