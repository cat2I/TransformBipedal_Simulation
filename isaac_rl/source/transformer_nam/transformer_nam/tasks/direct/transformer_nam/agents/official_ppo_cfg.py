"""Separate logs/checkpoints for the new robot and its new control interface."""
from isaaclab.utils.configclass import configclass
from .rsl_rl_ppo_cfg import TransformerWalkPPORunnerCfg


@configclass
class OfficialWalkPPORunnerCfg(TransformerWalkPPORunnerCfg):
    experiment_name = "officialdesign_walk"
    max_iterations = 3000
