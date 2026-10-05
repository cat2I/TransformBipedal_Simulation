# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TransformerWalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Transformer walking

    rsl-rl >= 4.0 bỏ ``policy = RslRlPpoActorCriticCfg(...)``, tách thành hai
    model riêng ``actor`` và ``critic``. Cờ ``empirical_normalization`` cũng
    chuyển thành ``obs_normalization`` trên từng model.

    Kiến trúc mạng và toàn bộ hyperparameter giữ NGUYÊN như cấu hình cũ
    ([256, 256, 128], elu, init_std=1.0, không chuẩn hoá observation) để các
    checkpoint đã train trước đây vẫn khớp shape.
    """

    # PPO Runner Config
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "transformer_walk"

    # Policy network — trước đây là policy.actor_hidden_dims
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    # Value network — trước đây là policy.critic_hidden_dims
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    # PPO algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
