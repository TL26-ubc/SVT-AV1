import argparse
import os
from ast import arg
from pathlib import Path

import numpy as np
from pyencoder.environment.av1_running_env import Av1RunningEnv
from pyencoder.environment.naive_env import Av1GymEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def prase_arg():
    parser = argparse.ArgumentParser(description="Train RL agent for video encoding")

    # Video and output
    parser.add_argument(
        "--file", help="Input video file", default="Data/akiyo_qcif.y4m"
    )
    parser.add_argument(
        "--output_dir", default="logs/", help="Output directory for models and logs"
    )

    # RL parameters
    parser.add_argument(
        "--algorithm", choices=["ppo", "dqn"], default="ppo", help="RL algorithm to use"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=50000, help="Total training timesteps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps per update (PPO only)",
    )

    # Environment parameters
    parser.add_argument(
        "--lambda_rd", type=float, default=0.1, help="Rate-distortion lambda"
    )
    parser.add_argument(
        "--max_frames", type=int, default=100, help="Maximum frames per episode"
    )

    # Training parameters
    parser.add_argument(
        "--eval_freq", type=int, default=5000, help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_freq", type=int, default=10000, help="Model save frequency"
    )

    args = parser.parse_args()

    return args

    # Create trainer and run pipeline
    # trainer = VideoEncodingTrainer(args)
    # trainer.run_complete_pipeline()


if __name__ == "__main__":

    args = prase_arg()

    # create envirnment
    base_output_path = Path(args.output_dir)
    env = Av1GymEnv(
        video_path=args.file,
        lambda_rd=args.lambda_rd,
    )
    env = Monitor(env, str(base_output_path))

    # eval_env = Av1GymEnv(
    #     video_path=args.file,
    #     lambda_rd=args.lambda_rd,
    # )
    # eval_env = Monitor(eval_env, base_output_path / "eval_monitor")

    model = None
    if args.algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            # tensorboard_log=str(base_output_path / "tensorboard"),
        )
    elif args.algorithm == "dqn":
        # Note: DQN doesn't directly support MultiDiscrete action spaces
        # You might need to flatten the action space or use a wrapper
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=args.batch_size,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            verbose=1,
            # tensorboard_log=str(base_output_path / "tensorboard"),
        )
        
    # training
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            # callback=callbacks,
            tb_log_name=f"{args.algorithm}_run",
        )

        # Save final model
        final_model_path = base_output_path / f"final_{args.algorithm}_model"
        model.save(str(final_model_path))
        print(f"Training completed! Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save current model
        interrupted_model_path = (
            base_output_path / f"interrupted_{args.algorithm}_model"
        )
        model.save(str(interrupted_model_path))
        print(f"Model saved to: {interrupted_model_path}")
