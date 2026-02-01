#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train BDX standing policy with RSL-RL.

Usage:
    # Train BDX standing
    python train_bdx_standing.py --task Isaac-Standing-BDX-v0 --num_envs 4096

    # Resume training
    python train_bdx_standing.py --task Isaac-Standing-BDX-v0 --num_envs 4096 --resume

    # Train with fewer environments (for testing)
    python train_bdx_standing.py --task Isaac-Standing-BDX-v0 --num_envs 512
"""

import argparse
from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Train BDX standing policy with RSL-RL.")

# Append AppLauncher args first
AppLauncher.add_app_launcher_args(parser)

# Add custom training arguments
parser.add_argument("--video", action="store_true", default=False, help="Record video during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Standing-BDX-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Name of run folder to load from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load.")
args_cli, hydra_args = parser.parse_known_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# Import BDX to register environments
import BDX  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from rsl_rl.runners import OnPolicyRunner
import isaaclab_rl.rsl_rl as rsl_rl
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg


def main():
    """Train with RSL-RL agent."""
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    
    # Load agent configuration
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # Override agent configuration if provided
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # Specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # Create log directory with timestamp
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap environment for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner from agent config
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # Write git state, command, and config to logger
    runner.add_git_repo_to_log(__file__)
    
    # Dump config to logger
    dump_yaml(os.path.join(runner.log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(runner.log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(runner.log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(runner.log_dir, "params", "agent.pkl"), agent_cfg)

    # Resume training
    if args_cli.resume:
        resume_path = get_checkpoint_path(
            runner.log_dir, run=args_cli.load_run, checkpoint=args_cli.checkpoint
        )
        print(f"[INFO] Resuming training from: {resume_path}")
        runner.load(resume_path)

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
