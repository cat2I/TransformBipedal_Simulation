# # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# """Script to play a checkpoint if an RL agent from RSL-RL."""

# """Launch Isaac Sim Simulator first."""

# import argparse
# import sys

# from isaaclab.app import AppLauncher

# # local imports
# import cli_args  # isort: skip

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument(
#     "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
# )
# parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# parser.add_argument(
#     "--use_pretrained_checkpoint",
#     action="store_true",
#     help="Use the pre-trained checkpoint from Nucleus.",
# )
# parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli, hydra_args = parser.parse_known_args()
# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # clear out sys.argv for Hydra
# sys.argv = [sys.argv[0]] + hydra_args

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import os
# import time

# import gymnasium as gym
# import torch
# from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# from isaaclab.envs import (
#     DirectMARLEnv,
#     DirectMARLEnvCfg,
#     DirectRLEnvCfg,
#     ManagerBasedRLEnvCfg,
#     multi_agent_to_single_agent,
# )
# from isaaclab.utils.assets import retrieve_file_path
# from isaaclab.utils.dict import print_dict

# from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
# from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.utils import get_checkpoint_path
# from isaaclab_tasks.utils.hydra import hydra_task_config

# import transformer_nam.tasks  # noqa: F401


# @hydra_task_config(args_cli.task, args_cli.agent)
# def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
#     """Play with RSL-RL agent."""
#     # grab task name for checkpoint path
#     task_name = args_cli.task.split(":")[-1]
#     train_task_name = task_name.replace("-Play", "")

#     # override configurations with non-hydra CLI arguments
#     agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
#     env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

#     # set the environment seed
#     # note: certain randomizations occur in the environment initialization so we set the seed here
#     env_cfg.seed = agent_cfg.seed
#     env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

#     # specify directory for logging experiments
#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     print(f"[INFO] Loading experiment from directory: {log_root_path}")
#     if args_cli.use_pretrained_checkpoint:
#         resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
#         if not resume_path:
#             print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
#             return
#     elif args_cli.checkpoint:
#         resume_path = retrieve_file_path(args_cli.checkpoint)
#     else:
#         resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

#     log_dir = os.path.dirname(resume_path)

#     # set the log directory for the environment (works for all environment types)
#     env_cfg.log_dir = log_dir

#     # create isaac environment
#     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

#     # convert to single-agent instance if required by the RL algorithm
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     # wrap for video recording
#     if args_cli.video:
#         video_kwargs = {
#             "video_folder": os.path.join(log_dir, "videos", "play"),
#             "step_trigger": lambda step: step == 0,
#             "video_length": args_cli.video_length,
#             "disable_logger": True,
#         }
#         print("[INFO] Recording videos during training.")
#         print_dict(video_kwargs, nesting=4)
#         env = gym.wrappers.RecordVideo(env, **video_kwargs)

#     # wrap around environment for rsl-rl
#     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

#     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#     # load previously trained model
#     if agent_cfg.class_name == "OnPolicyRunner":
#         runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
#     elif agent_cfg.class_name == "DistillationRunner":
#         runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
#     else:
#         raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
#     runner.load(resume_path)

#     # obtain the trained policy for inference
#     policy = runner.get_inference_policy(device=env.unwrapped.device)

#     # extract the neural network module
#     # we do this in a try-except to maintain backwards compatibility.
#     try:
#         # version 2.3 onwards
#         policy_nn = runner.alg.policy
#     except AttributeError:
#         # version 2.2 and below
#         policy_nn = runner.alg.actor_critic

#     # extract the normalizer
#     if hasattr(policy_nn, "actor_obs_normalizer"):
#         normalizer = policy_nn.actor_obs_normalizer
#     elif hasattr(policy_nn, "student_obs_normalizer"):
#         normalizer = policy_nn.student_obs_normalizer
#     else:
#         normalizer = None

#     # export policy to onnx/jit
#     export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
#     export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
#     export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

#     dt = env.unwrapped.step_dt

#     # reset environment
#     obs = env.get_observations()
#     timestep = 0
#     # simulate environment
#     while simulation_app.is_running():
#         start_time = time.time()
#         # run everything in inference mode
#         with torch.inference_mode():
#             # agent stepping
#             actions = policy(obs)
            
#             # ✅ LOGGING: Display observations and actions
#             if timestep % 2 == 0:
#                 # Lấy góc thực tế đang được apply (cmd_actions của env)
#                 env_internal = env.unwrapped  # vì RslRlVecEnvWrapper
#                 real_angles = env_internal.cmd_actions[0].cpu().numpy()  # 6 góc hiện tại (degree)
                
#                 # Hoặc nếu muốn in noisy_act (có noise + backlash):
#                 # real_angles = env_internal.noisy_act[0].cpu().numpy()
                
#                 roll  = obs["policy"][0][15].item()   # tùy version 44D, bạn có thể chỉnh index
#                 pitch = obs["policy"][0][16].item()
#                 gx    = obs["policy"][0][17].item()
#                 gy    = obs["policy"][0][18].item()
#                 gz    = obs["policy"][0][19].item()
                
#                 print(f"Step {timestep:5d} | "
#                     f"Orient: roll={roll:+7.4f}rad pitch={pitch:+7.4f}rad | "
#                     f"Gyro: gx={gx:+7.4f} gy={gy:+7.4f} gz={gz:+7.4f} | "
#                     f"Obs: {obs['policy'][0].shape[0]}D | "
#                     f"REAL Angles(°): {[f'{a:+6.1f}' for a in real_angles]}")
            
#             # env stepping
#             obs, _, dones, _ = env.step(actions)
#             # reset recurrent states for episodes that have terminated
#             policy_nn.reset(dones)
#             # Increment timestep
#             timestep += 1
#         if args_cli.video:
#             # Exit the play loop after recording one video
#             if timestep == args_cli.video_length:
#                 break

#         # time delay for real-time evaluation
#         sleep_time = dt - (time.time() - start_time)
#         if args_cli.real_time and sleep_time > 0:
#             time.sleep(sleep_time)

#     # close the simulator
#     env.close()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()


# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of TransformerTwistMarch-v0 (10 DOF)."""

import argparse
import sys
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play TransformerTwistMarch-v0 (10 DOF).")
parser.add_argument("--video",             action="store_true", default=False)
parser.add_argument("--video_length",      type=int, default=200)
parser.add_argument("--disable_fabric",    action="store_true", default=False)
parser.add_argument("--num_envs",          type=int, default=None)
parser.add_argument("--task",              type=str, default="TransformerTwistMarch-v0")
parser.add_argument("--agent",             type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed",              type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time",         action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# IsaacLab 3.0 đảo ngược mặc định so với 2.x: KHÔNG truyền '--viz' là chạy headless
# (xem app_launcher.py::_resolve_headless_settings — không có visualizer nào được
# chọn thì nó tự ép self._headless = True). Nghĩa là bỏ '--headless' không còn làm
# hiện cửa sổ nữa, phải nói rõ '--viz kit'.
#
# Play là để NHÌN robot, nên mặc định bật luôn visualizer Kit. Hai đường thoát vẫn
# nguyên vẹn, vì nhánh này chỉ chạy khi người dùng không nói gì:
#   --headless      -> chạy ngầm như cũ
#   --viz <tên>     -> tự chọn visualizer, và real_time giữ False (chạy nhanh tối đa)
# Dùng 'visualizer_explicit' chứ KHÔNG dùng 'visualizer is None': '--viz none' cũng
# cho ra None (xem AppLauncher._parse_visualizer_csv), nên kiểm tra None sẽ hiểu nhầm
# "người dùng ép tắt" thành "người dùng không nói gì" rồi bật cửa sổ ngược ý họ.
# ExplicitAction đặt cờ '<dest>_explicit' đúng khi cờ được gõ ra, phân biệt được hai ca.
if not args_cli.headless and not getattr(args_cli, "visualizer_explicit", False):
    args_cli.visualizer = ["kit"]   # mở cửa sổ Isaac Sim
    args_cli.real_time = True       # chạy đúng tốc độ thật, không thì tua quá nhanh
    # KHÔNG đặt rendering_mode ở đây. Đã thử '--rendering_mode performance' cho GPU
    # 6GB và nó làm viewport ĐEN HOÀN TOÀN — mất cả lưới sàn lẫn robot, trong khi sim
    # vẫn chạy và vẫn in log bình thường, nên rất dễ tưởng là lỗi khác. Để mặc định
    # của Kit thì hiện đủ. Muốn thử lại thì truyền tay '--rendering_mode <mode>'.

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, time, math
import importlib.metadata as metadata
import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg, RslRlVecEnvWrapper,
    export_policy_as_jit, export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import transformer_nam.tasks  # noqa: F401


# ── Obs layout (62D) — khớp TransformerTwistMarchEnv ────────────────────
#
# Joint order trong env (10 khớp revolute) — XEN KẼ trái/phải, không phải trái hết
# rồi mới tới phải. Đây là thứ tự Isaac Sim thật sự nạp, đọc từ bảng "Simulation Joint
# Information" mà nó in ra lúc khởi động:
#   [0] Bubleft   [1] Bubright   [2] Hipleft   [3] Hipright   [4] Twistleft
#   [5] Twistright [6] Kneeleft  [7] Kneeright [8] Footleft   [9] Footright
#
# Đừng viết cứng mấy chỉ số này ở chỗ khác — dùng joint_index() bên dưới. Bản trước
# giả định "trái hết rồi tới phải" nên đọc nhầm index 2/7 (Hipleft/Kneeright) mà vẫn
# in nhãn TwL/TwR, khiến log trông hợp lý nhưng sai khớp.
#
# Obs slices:
#   [ 0: 20)  imu_hist   — 4 frames × (roll, pitch, gx, gy, gz) = 4×5 = 20D
#   [20: 60)  jpos_hist  — 4 frames × 10 joints normalized        = 4×10 = 40D
#   [60: 62)  twist_prog — [twist_L_progress, twist_R_progress]   2D
#                          (1.0 = start/90°, 0.0 = done/0°)
#
# Trong mỗi frame của imu_hist (5D):
#   [0] roll   [1] pitch   [2] gx   [3] gy   [4] gz
# Frame 0 = oldest, frame 3 = newest → frame 3 bắt đầu ở offset 15

OBS_IMU_START   =  0    # 20D imu history
OBS_JPOS_START  = 20    # 40D joint pos history (4 frames × 10 joints)
OBS_TWIST_L     = 60    # twist_L progress (1=start/90°, 0=done/0°)
OBS_TWIST_R     = 61    # twist_R progress

# Offset của frame cuối (newest) trong imu_hist
_IMU_LAST_FRAME = OBS_IMU_START + 3 * 5   # = 15

# Offset của frame cuối trong jpos_hist
_JPOS_LAST_FRAME = OBS_JPOS_START + 3 * 10  # = 50

def joint_index(robot, name: str) -> int | None:
    """Chỉ số của khớp theo TÊN, hoặc None nếu asset không có khớp đó.

    Tra theo tên chứ không viết cứng số, vì thứ tự khớp do file USD quyết định — đổi
    asset là số đổi theo, mà kiểu sai đó hoàn toàn im lặng: log vẫn in ra số đẹp, chỉ
    là của khớp khác. Đã dính đúng một lần (index 2/7 hoá ra là Hipleft/Kneeright).
    """
    names = getattr(robot, "joint_names", None)
    if not names:
        return None
    try:
        return list(names).index(name)
    except ValueError:
        return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    task_name       = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # gỡ các trường cấu hình đã deprecated trước khi đưa vào rsl-rl
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None
        else env_cfg.scene.num_envs
    )
    env_cfg.seed       = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None
        else env_cfg.sim.device
    )

    # ── Load checkpoint ──────────────────────────────────────────────────
    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    print(f"[INFO] Loading experiment from: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint found.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir         = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # ── Build env ────────────────────────────────────────────────────────
    env = gym.make(
        args_cli.task, cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder":  os.path.join(log_dir, "videos", "play"),
            "step_trigger":  lambda step: step == 0,
            "video_length":  args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Load policy ──────────────────────────────────────────────────────
    print(f"[INFO] Loading model checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Runner không hỗ trợ: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # rsl-rl >= 4.0 tách actor/critic thành hai model rời, nên không còn
    # ``alg.policy`` lẫn ``alg.actor_critic``; API mới là ``alg.get_policy()``.
    if hasattr(runner.alg, "get_policy"):
        policy_nn = runner.alg.get_policy()
    elif hasattr(runner.alg, "policy"):
        policy_nn = runner.alg.policy
    else:
        policy_nn = runner.alg.actor_critic

    # ── Export model ─────────────────────────────────────────────────────
    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if hasattr(runner, "export_policy_to_jit"):
        # rsl-rl >= 4.0: runner tự rút actor và normalizer đúng cách. Không dùng
        # export_policy_as_jit của isaaclab_rl vì hàm đó đòi ``policy.actor``,
        # tức module ActorCritic gộp của bản cũ — MLPModel mới không có.
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    else:
        normalizer = (
            getattr(policy_nn, "actor_obs_normalizer", None)
            or getattr(policy_nn, "student_obs_normalizer", None)
        )
        export_policy_as_jit(
            policy_nn, normalizer=normalizer,
            path=export_dir, filename="policy.pt",
        )
        export_policy_as_onnx(
            policy_nn, normalizer=normalizer,
            path=export_dir, filename="policy.onnx",
        )

    # ── Play loop ────────────────────────────────────────────────────────
    dt       = env.unwrapped.step_dt
    obs      = env.get_observations()
    timestep = 0
    e        = env.unwrapped   # TransformerTwistMarchEnv instance

    print("\n" + "="*75)
    print(
        f"PLAYING {args_cli.task}  |  "
        f"Obs {env.unwrapped.cfg.observation_space.shape[0]}D  |  "
        f"Act {env.unwrapped.cfg.action_space.shape[0]}D"
    )
    print(f"Checkpoint: {resume_path}")

    # Tra chỉ số hai khớp xoay MỘT LẦN, theo tên. Thiếu tên thì in cảnh báo rồi bỏ hẳn
    # hai cột đó — thà không có số còn hơn có số của khớp khác mà vẫn dán nhãn TwL/TwR.
    twist_l_idx = joint_index(e.robot, "Twistleft_joint")
    twist_r_idx = joint_index(e.robot, "Twistright_joint")
    if twist_l_idx is None or twist_r_idx is None:
        print(
            "[CẢNH BÁO] Asset không có Twistleft_joint/Twistright_joint — bỏ hai cột "
            f"TwL/TwR. Các khớp đang có: {list(getattr(e.robot, 'joint_names', []))}"
        )
    else:
        print(f"Khớp xoay: Twistleft=[{twist_l_idx}]  Twistright=[{twist_r_idx}]")
    print("="*75 + "\n")

    while simulation_app.is_running():
        t0 = time.time()

        with torch.inference_mode():
            actions = policy(obs)

            # ── Log mỗi 2 step ───────────────────────────────────────
            # Script này gốc viết riêng cho TransformerTwistMarch-v0 (62D obs,
            # có buffer lệnh twist và static_timer). Các task Walk10DOF đang
            # dùng chỉ có 60D obs và KHÔNG có những thuộc tính đó, nên phần nào
            # thiếu thì bỏ qua thay vì ném IndexError/AttributeError.
            if timestep % 2 == 0:
                p = obs["policy"][0]

                # IMU frame mới nhất (frame index 3, offset 15)
                roll  = p[_IMU_LAST_FRAME + 0].item()
                pitch = p[_IMU_LAST_FRAME + 1].item()
                gz    = p[_IMU_LAST_FRAME + 4].item()

                # Twist thực tế từ env (deg) — đọc joint pos trực tiếp
                if twist_l_idx is not None:
                    real_twist_L = math.degrees(
                        e.robot.data.joint_pos[0, twist_l_idx].item()
                    )
                    real_twist_R = math.degrees(
                        e.robot.data.joint_pos[0, twist_r_idx].item()
                    )

                # Chiều cao robot
                h = e.robot.data.root_pos_w[0, 2].item()

                # Contact / air time
                air_time = e.scene.sensors["contact"].data.current_air_time[0]
                air_L    = air_time[0].item()
                air_R    = air_time[1].item()
                thr = getattr(e.cfg, "air_time_threshold", 0.05)
                foot_on_L = air_L < thr
                foot_on_R = air_R < thr

                line = (
                    f"[{timestep:5d}] "
                    f"roll={roll:+5.3f} pitch={pitch:+5.3f} gz={gz:+5.3f} "
                    f"h={h:.3f}m"
                )
                if twist_l_idx is not None:
                    line += f" | TwL={real_twist_L:+6.1f}° TwR={real_twist_R:+6.1f}°"

                # Lệnh twist — chỉ TwistMarch mới có buffer cmd_deg
                cmd_deg = getattr(e, "cmd_deg", None)
                if cmd_deg is not None and twist_l_idx is not None:
                    line += (
                        f" (cmd L={cmd_deg[0, twist_l_idx].item():+6.1f}° "
                        f"R={cmd_deg[0, twist_r_idx].item():+6.1f}°)"
                    )

                # Tiến độ twist — nằm ở obs[60], obs[61], chỉ có khi obs đủ 62D
                has_prog = p.shape[0] > OBS_TWIST_R
                if has_prog:
                    line += (
                        f" prog=[{p[OBS_TWIST_L].item():.2f},"
                        f"{p[OBS_TWIST_R].item():.2f}]"
                    )

                line += (
                    f" | air L={air_L:.2f}s R={air_R:.2f}s "
                    f"contact L={'Y' if foot_on_L else 'n'}"
                    f"R={'Y' if foot_on_R else 'n'}"
                )

                # Static timer — chỉ TwistMarch mới có
                static_timer = getattr(e, "static_timer", None)
                static_t = static_timer[0].item() if static_timer is not None else None
                if static_t is not None:
                    line += f" static={static_t:.2f}s"

                print(line)

                # Cảnh báo — chỉ có nghĩa với nhiệm vụ twist, và chỉ khi đọc được khớp
                if has_prog and twist_l_idx is not None:
                    if timestep > 500 and (abs(real_twist_L) > 45 or abs(real_twist_R) > 45):
                        print(f"  ⚠  Twist chưa về 0° sau {timestep} step")
                    if abs(real_twist_L) < 5.0 and abs(real_twist_R) < 5.0:
                        print(
                            f"  ✓  CẢ 2 CHÂN XOAY XONG! "
                            f"TwL={real_twist_L:.1f}° TwR={real_twist_R:.1f}°"
                        )
                if static_t is not None:
                    limit = getattr(e.cfg, "static_penalty_threshold", None)
                    if limit is not None and static_t > limit + 0.5:
                        print(f"  ⚠  Robot đứng yên quá lâu: {static_t:.2f}s")

            obs, _, dones, _ = env.step(actions)
            # MLP thuần của rsl-rl >= 4.0 không có .reset() — chỉ policy hồi quy
            # (RNN/LSTM) mới cần xoá hidden state khi env kết thúc episode.
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)
            timestep += 1

        if args_cli.video and timestep == args_cli.video_length:
            break

        elapsed   = time.time() - t0
        remaining = dt - elapsed
        if args_cli.real_time and remaining > 0:
            time.sleep(remaining)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()