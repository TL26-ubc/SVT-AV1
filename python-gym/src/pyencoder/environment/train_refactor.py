import argparse
from pathlib import Path

from pyencoder.environment.naive_env import Av1GymEnv
from pyencoder.states.naive import NaiveState
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor


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
        "--total_iteration", type=int, default=50, help="Total training loop iterations, number of times the environment is reset"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=-1,
        help="Number of steps per update (PPO only), should match number of frames in the video, -1 would search for the video length",
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

    parser.add_argument(
        "--disable_observation_normalization", 
        action="store_true", 
        help="Disable observation state normalization"
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
    gyn_env = Av1GymEnv(
        video_path=args.file,
        output_dir=base_output_path,
        lambda_rd=args.lambda_rd,
        state=NaiveState,
    )
    
    env = Monitor(gyn_env, str(base_output_path / "monitor"))
    if args.n_steps == -1:
        # Automatically determine n_steps based on video length
        video_length = gyn_env.num_frames
        args.n_steps = video_length if video_length > 0 else 1000  # Fallback to 1000 if length is unknown

    model = None
    match args.algorithm.lower():
        case "ppo":
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
            )
        case "dqn":
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
            )
        case other:
            raise ValueError(f"Unsupported algorithm '{other}'. Choose either 'ppo' or 'dqn'.")
        
    total_timesteps = args.total_iteration * gyn_env.num_frames

    # training
    try:
        model.learn(
            total_timesteps=total_timesteps,
            # callback=callbacks,
            tb_log_name=f"{args.algorithm}_run",
        )

        # Save final model
        final_model_path = base_output_path / f"final_{args.algorithm}_model"
        model.save(str(final_model_path))
        gyn_env.save_bitstream_to_file(
            str(base_output_path / "final_encoder_video.ivf")
        )
        print(f"Training completed! Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save current model
        interrupted_model_path = (
            base_output_path / f"interrupted_{args.algorithm}_model"
        )
        model.save(str(interrupted_model_path))
        gyn_env.save_bitstream_to_file(
            str(base_output_path / "interrupted_encoder_video.ivf"),
            interrupt=True
        )
        print(f"Model saved to: {interrupted_model_path}")