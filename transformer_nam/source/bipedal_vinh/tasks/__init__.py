import gymnasium as gym

gym.register(
    id="VinhRobot-10DOF-v0",  # Tên môi trường độc lập của bạn
    entry_point="bipedal_vinh.tasks.vinh10dof_env:TransformerWalk10DOFEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "bipedal_vinh.tasks.vinh10dof_env:TransformerWalk10DOFEnvCfg",
        # Phần thuật toán RL: Kế thừa từ thư mục cũ
        "rsl_rl_cfg_entry_point": "transformer_nam.tasks.direct.transformer_nam.agents.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",
    },
)
