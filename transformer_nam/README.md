# Transformer Bipedal Walking - Isaac Lab RL Training

Reinforcement learning environment for training a bipedal robot (Transformer) using Isaac Lab with 6 DOF locomotion control.

## Demo

![Transformer Walking](0319.gif)

*Real robot walking with learned policy (0.5x speed)*

---

## Overview

**Robot Configuration:**
- **DOF:** 6 (Hip_L, Hip_R, Knee_L, Knee_R, Ankle_L, Ankle_R)
- **Sensors:** IMU (Baselink), Contact sensors (2 feet)
- **Feet:** PETG plastic
- **Ground:** Polished wooden floor

**Observation Space:** 44D
- 20D IMU history: 4 timesteps × (roll, pitch, gyro_x, gyro_y, gyro_z)
- 24D action history: 4 timesteps × 6 joints

**Action Space:** 6D continuous [-3.0, +3.0] normalized

## Physics Parameters

```yaml
Ground:
  static_friction: 0.8
  dynamic_friction: 0.4
  
Joint Control:
  friction_range: [0.3, 0.5] (randomized per episode)
  backlash: 2.5° (gear deadband)
  actuator_delay: 1-4 steps (5-20ms random per episode)
  
Control:
  dt: 0.005s
  decimation: 10
  frequency: 200Hz
  parallel_envs: 512
```

## Installation

1. **Install Isaac Lab**
   ```bash
   # Follow: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
   conda activate env_isaaclab
   ```

2. **Install Extensions**
   ```bash
   python -m pip install -e source/transformer_nam
   ```

3. **Verify Installation**
   ```bash
   python scripts/list_envs.py | grep Transformer-Walk
   ```

## Training

### Start Training
```bash
# Headless mode, 512 environments, 350 iterations
~/IsaacLab/isaac-sim/python.sh scripts/rsl_rl/train.py \
    --task Transformer-Walk-Direct-v0 \
    --num_envs 512 \
    --headless \
    --max_iterations 350
```

### Monitor Training
```bash
tensorboard --logdir=logs/rsl_rl/transformer_walk --port=6006
```

## Inference

### Run Trained Policy
```bash
~/IsaacLab/isaac-sim/python.sh scripts/rsl_rl/play.py \
    --task Transformer-Walk-Direct-v0 \
    --num_envs 1 \
    --load_run logs/rsl_rl/transformer_walk/<checkpoint>
```

### View Logs
```
[Step N] Episode M
  Orientation reward: X.XXX
  Height reward: X.XXX
  Velocity reward: X.XXX
  Feet height reward: X.XXX
  TOTAL: X.XXX
```

## Reward Function

7-component weighted reward:

| Component | Weight | Description |
|-----------|--------|-------------|
| Orientation | 1.0 | Keep roll/pitch < 0.95 rad |
| Height | 1.0 | Maintain 0.392m ± 0.3m |
| Joint Position | 1.0 | Stay ±5° from default |
| Sigmoid Bonus | 0 | Extra for good control |
| Feet Clearance | 2.0 | Swing feet 3cm above ground |
| Velocity | 1.0 | Forward motion (minimize lateral drift) |
| Deviation | 1.0 | Stay within ±0.3m lateral |

## Real Robot Integration

### Export 1000-Step Trajectory
```bash
python export_trajectory.py \
    --checkpoint logs/rsl_rl/transformer_walk/.../model.pt \
    --num_steps 1000 \
    --output trajectory.json
```

**JSON Format:**
```json
{
  "metadata": {
    "checkpoint": "...",
    "joints": ["Hip_L", "Hip_R", "Knee_L", "Knee_R", "Ankle_L", "Ankle_R"],
    "servo_limits": {...},
    "duration_s": 5.0
  },
  "trajectory": [
    {
      "step": 0,
      "time_s": 0.0,
      "angles_deg": [25, 25, -50, -50, 25, 25],
      "imu_data": {...},
      "raw_actions": [-0.1, 0.2, -0.5, 0.1, 0.0, -0.2]
    },
    ...
  ]
}
```

### Playback on Real Robot
```python
from trajectory_playback import RealRobotPlayback

playback = RealRobotPlayback('trajectory.json')
playback.play(speed_factor=1.0)  # 1.0x = ~5s walk
```

## Project Structure

```
transformer_nam/
├── source/transformer_nam/
│   └── transformer_nam/
│       ├── tasks/
│       │   └── direct/transformer_nam/
│       │       ├── transformer_nam_env.py     # RL environment
│       │       ├── transformer_config.py      # Robot + physics config
│       │       └── __init__.py
│       └── ui/
│           └── ui_extension_example.py
├── scripts/
│   ├── play.py                   # Inference + logging
│   ├── export_trajectory.py      # Export to JSON
│   └── trajectory_playback.py    # Replay on real robot
├── asset/
│   └── *.usd                     # USD simulation models
└── README.md
```

## Key Features

### Backlash Simulation
- Gear deadband: 2.5° when direction changes
- Represents mechanical play in servo gears
- Reduces unrealistic sharp movements

### Actuator Delay
- Random 1-4 steps per episode (5-20ms)
- Each step = 5ms (dt=0.005s)
- Simulates servo response latency

### Joint Friction Randomization
- Range: 0.3-0.5 coefficient
- Randomized each reset
- Improves sim-to-real transfer

### IMU Noise
- Orientation: Gaussian ±0.015 rad
- Gyroscope: Gaussian ±0.01 rad/s

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `RuntimeError: shape mismatch` | Verify `num_actions=6`, `num_observations=44` in config |
| Training unstable | Check friction range, adjust reward weights |
| Slow convergence | Increase `num_envs` (512+), tune learning rate |
| USD files missing on GitHub | Remove `**/*.usd` from `.gitignore` |
| Feet slip on ground | Decrease `dynamic_friction` to 0.3-0.4 |

## Development

### IDE Setup (VSCode)
1. Press `Ctrl+Shift+P` → `Tasks: Run Task` → `setup_python_env`
2. Enter path to Isaac Sim installation
3. Creates `.python.env` for indexing

### Code Formatting
```bash
pip install pre-commit
pre-commit run --all-files
```

## References

- [Isaac Lab Documentation](https://docs.omniverse.nvidia.com/isaaclab/)
- [RSL-RL Repository](https://github.com/leggedrobotics/rsl_rl)
- [IsaacSim](https://docs.omniverse.nvidia.com/isaacsim/)

## License

[Specify your license here]

## Authors

- Transformer Bipedal Project Team