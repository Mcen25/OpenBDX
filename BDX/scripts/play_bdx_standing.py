#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a trained BDX standing policy.

Usage:
    # Play the latest checkpoint
    python play_bdx_standing.py --task Isaac-Standing-BDX-Play-v0

    # Play a specific checkpoint
    python play_bdx_standing.py --task Isaac-Standing-BDX-Play-v0 --checkpoint /path/to/model.pt

    # Play with fewer environments
    python play_bdx_standing.py --task Isaac-Standing-BDX-Play-v0 --num_envs 16
"""

import argparse
from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Play BDX standing policy with RSL-RL.")

# Append AppLauncher args first
AppLauncher.add_app_launcher_args(parser)

# Add custom play arguments
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Standing-BDX-Play-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
args_cli = parser.parse_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

# Import BDX to register environments
import BDX  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv

from rsl_rl.runners import OnPolicyRunner
import isaaclab_rl.rsl_rl as rsl_rl
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Load the agent configuration
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load checkpoint
    if args_cli.checkpoint:
        checkpoint_path = args_cli.checkpoint
    else:
        # Load the latest checkpoint
        checkpoint_path = get_checkpoint_path(agent_cfg.logger.log_dir)
    
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    runner.load(checkpoint_path)

    # Reset environment
    obs, _ = env.reset()
    
    # Simulate
    with torch.inference_mode():
        while simulation_app.is_running():
            # Get actions from policy
            actions = runner.get_inference_actions(obs)
            
            # Apply actions
            obs, _, _, _, _ = env.step(actions)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
