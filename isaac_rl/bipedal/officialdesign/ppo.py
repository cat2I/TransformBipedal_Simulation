"""Separate logs/checkpoints for the new robot and its new control interface."""
from isaaclab.utils.configclass import configclass
from .._shared.ppo import TransformerWalkPPORunnerCfg


@configclass
class OfficialWalkPPORunnerCfg(TransformerWalkPPORunnerCfg):
    experiment_name = "officialdesign"
    max_iterations = 3000
