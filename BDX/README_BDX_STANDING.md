# BDX Bipedal Robot - Standing Policy Training

This directory contains the complete setup for training a standing/balancing policy for the BDX bipedal robot using Isaac Lab and RSL-RL.

## Overview

The BDX standing task trains the robot to:
- **Stand upright** and maintain balance
- **Minimize drift** from the starting position
- **Maintain target height** (~60cm)
- **Resist external disturbances** (random pushes)
- **Minimize energy consumption** (torque and power penalties)

## Files Structure

```
BDX/
├── source/BDX/BDX/tasks/manager_based/bdx/
│   ├── bdx_robot_cfg.py              # Robot asset configuration
│   ├── bdx_standing_env_cfg.py       # Standing environment configuration
│   ├── bdx_env_cfg.py                # Template cartpole example
│   ├── agents/
│   │   ├── bdx_standing_ppo_cfg.py   # PPO training configuration
│   │   └── rsl_rl_ppo_cfg.py         # Template PPO config
│   └── __init__.py                   # Environment registration
├── scripts/
│   ├── train_bdx_standing.py         # Training script
│   └── play_bdx_standing.py          # Inference/play script
└── README_BDX_STANDING.md            # This file
```

## Robot Configuration

### Hardware Specs (Feetech ST-3215-C044 Servos)
- **Stall Torque**: 2.686 N·m (27.4 kg·cm @ 7.4V)
- **No-Load Speed**: 9.01 rad/s (86 RPM @ 7.4V)
- **Gear Ratio**: 1:191
- **Control**: Position mode with PID

### Joint Configuration
- 10 joints total (5 per leg)
- Left leg: `joint_1_left` through `joint_5_left`
- Right leg: `joint_1_right` through `joint_5_right`

## Training

### Basic Training

```bash
# Train with default settings (4096 environments)
python BDX/scripts/train_bdx_standing.py --task Isaac-Standing-BDX-v0

# Train with fewer environments (for testing/debugging)
python BDX/scripts/train_bdx_standing.py --task Isaac-Standing-BDX-v0 --num_envs 512

# Resume training from checkpoint
python BDX/scripts/train_bdx_standing.py --task Isaac-Standing-BDX-v0 --resume

# Train with video recording
python BDX/scripts/train_bdx_standing.py --task Isaac-Standing-BDX-v0 --video
```

### Training Configuration

- **Episode Length**: 20 seconds
- **Control Frequency**: 50 Hz (decimation=4, sim_dt=0.005)
- **Number of Environments**: 4096 (default)
- **Max Iterations**: 1500
- **Policy Network**: [256, 128, 64] with ELU activation
- **Value Network**: [256, 128, 64] with ELU activation

### Training Progress

Logs are saved to: `logs/rsl_rl/bdx_standing/<timestamp>/`

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/bdx_standing
```

## Playing/Inference

### Play Trained Policy

```bash
# Play latest checkpoint
python BDX/scripts/play_bdx_standing.py --task Isaac-Standing-BDX-Play-v0

# Play specific checkpoint
python BDX/scripts/play_bdx_standing.py --task Isaac-Standing-BDX-Play-v0 \
    --checkpoint logs/rsl_rl/bdx_standing/*/model_1000.pt

# Play with fewer environments
python BDX/scripts/play_bdx_standing.py --task Isaac-Standing-BDX-Play-v0 --num_envs 4
```

## Reward Terms

The standing policy is shaped by multiple reward terms:

### Primary Rewards
| Reward | Weight | Description |
|--------|--------|-------------|
| `upright_reward` | 2.0 | Bonus for staying upright (>0.93 threshold) |
| `base_position_tracking` | 1.5 | Penalize drift from start position |
| `base_height_target` | 1.0 | Maintain 0.6m target height |

### Motion Penalties
| Penalty | Weight | Description |
|---------|--------|-------------|
| `lin_vel_z_l2` | -2.0 | Minimize vertical velocity |
| `ang_vel_xy_l2` | -0.5 | Minimize pitch/roll velocity |
| `ang_vel_z_l2` | -0.5 | Minimize yaw velocity |
| `joint_vel_l2` | -1e-4 | Penalize joint velocities |
| `joint_accel_l2` | -2.5e-7 | Penalize joint accelerations |

### Energy & Smoothness
| Penalty | Weight | Description |
|---------|--------|-------------|
| `applied_torque_limits` | -0.01 | Penalize high torques |
| `power` | -0.0002 | Penalize power consumption |
| `action_rate_l2` | -0.01 | Encourage smooth actions |
| `joint_deviation_l1` | -0.2 | Stay close to default pose |

### Contact Penalties
| Penalty | Weight | Description |
|---------|--------|-------------|
| `undesired_contacts` | -1.0 | Penalize non-foot contacts |

## Observations

The policy observes (total ~33 dims):
- **Base state** (9): linear velocity, angular velocity, projected gravity
- **Joint state** (20): 10 joint positions + 10 joint velocities
- **Previous actions** (10): Last commanded joint positions

All observations include realistic noise modeling.

## Randomization

### Reset Events
- Base position: ±20cm XY, 55-65cm Z
- Base velocity: ±0.1 m/s linear, ±0.1 rad/s angular
- Joint positions: 80-120% of default
- Joint velocities: ±0.5 rad/s

### Interval Events
- **External pushes**: Random 0.5 m/s pushes every 10-15 seconds

## Termination Conditions

Training episodes terminate when:
1. **Time out**: 20 seconds elapsed
2. **Illegal contact**: Base/torso touches ground

## Next Steps

After training a standing policy, you can:

1. **Increase difficulty**: Add terrain randomization, increase push forces
2. **Transfer to walking**: Use standing policy as initialization for velocity tracking
3. **Sim-to-real**: Deploy policy on real BDX hardware with domain randomization

## Troubleshooting

### GPU Memory Issues
If you see PhysX GPU memory errors, reduce the number of environments:
```bash
python BDX/scripts/train_bdx_standing.py --task Isaac-Standing-BDX-v0 --num_envs 2048
```

### Robot Falls Immediately
- Check URDF is properly converted to USD with correct joint limits
- Verify motor torque limits match hardware specs
- Ensure base starts at proper height (0.6m)

### Training Unstable
- Reduce learning rate: Modify `bdx_standing_ppo_cfg.py`
- Increase damping: Modify `bdx_robot_cfg.py`
- Adjust reward weights: Modify `bdx_standing_env_cfg.py`

## Related Documentation

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL Documentation](https://github.com/leggedrobotics/rsl_rl)
- Motor Specs: `/Assets/my-robot/101090141_Feetech_ST-3215-C044_Datasheet.pdf`
