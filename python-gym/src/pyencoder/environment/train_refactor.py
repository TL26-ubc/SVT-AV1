import argparse
from pathlib import Path

from pyencoder.environment.naive_env import Av1GymEnv
from pyencoder.states.naive import NaiveState
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


def create_lr_schedule(scheduler_type: str, initial_lr: float, **kwargs):
    """Create SB3-compatible learning rate schedule function"""
    
    if scheduler_type == "none":
        return initial_lr
    
    elif scheduler_type == "linear":
        final_lr_ratio = kwargs.get('final_lr_ratio', 0.1)
        def linear_schedule(progress_remaining):
            # progress_remaining goes from 1.0 to 0.0
            return initial_lr * (final_lr_ratio + (1.0 - final_lr_ratio) * progress_remaining)
        return linear_schedule
    
    elif scheduler_type == "cosine":
        min_lr_ratio = kwargs.get('cosine_min_ratio', 0.0)
        def cosine_schedule(progress_remaining):
            # Cosine annealing: starts at initial_lr, goes to min_lr_ratio * initial_lr
            progress = 1.0 - progress_remaining  # Convert to progress from 0 to 1
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return initial_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)
        return cosine_schedule
    
    elif scheduler_type == "exponential":
        final_lr_ratio = kwargs.get('final_lr_ratio', 0.1)
        # Calculate decay rate to reach final_lr_ratio at the end
        decay_rate = np.log(final_lr_ratio)
        def exponential_schedule(progress_remaining):
            progress = 1.0 - progress_remaining
            return initial_lr * np.exp(decay_rate * progress)
        return exponential_schedule
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WandbCallback(BaseCallback):
    def __init__(self, lr_log_freq=50, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_psnr_y = []
        self.episode_psnr_cb = []
        self.episode_psnr_cr = []
        self.episode_bitrates = []
        self.step_count = 0
        self.last_logged_lr = None
        self.lr_log_freq = lr_log_freq

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Log step-level metrics
        step_metrics = {}
        
        # Add model training metrics if available
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            if hasattr(self.model.logger, 'name_to_value'):
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float, np.number)):
                        step_metrics[f"train/{key}"] = value
        
        # Get learning rate - now SB3 will handle the scheduling properly
        current_lr = None
        
        # Get from PyTorch optimizer (this will show the actual LR being used)
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            optimizer = self.model.policy.optimizer
            current_lr = optimizer.param_groups[0]['lr']
        
        # Get SB3's scheduled learning rate
        sb3_lr = None
        if hasattr(self.model, 'learning_rate'):
            if callable(self.model.learning_rate):
                sb3_lr = self.model.learning_rate(self.model._current_progress_remaining)
            else:
                sb3_lr = self.model.learning_rate
        
        # Log learning rates
        if (self.last_logged_lr is None or 
            (current_lr is not None and abs(current_lr - self.last_logged_lr) > 1e-9) or 
            self.step_count % self.lr_log_freq == 0):
            
            if current_lr is not None:
                step_metrics["train/learning_rate"] = current_lr
                step_metrics["train/pytorch_lr"] = current_lr
            if sb3_lr is not None:
                step_metrics["train/sb3_lr"] = sb3_lr
            
            # Debug print when learning rate changes
            if self.verbose > 0 and current_lr is not None:
                if self.last_logged_lr is not None and abs(current_lr - self.last_logged_lr) > 1e-9:
                    print(f"Step {self.step_count}: Learning Rate changed from {self.last_logged_lr:.8f} to {current_lr:.8f}")
                    if sb3_lr is not None:
                        print(f"  SB3 scheduled LR: {sb3_lr:.8f}, PyTorch actual LR: {current_lr:.8f}")
            
            if current_lr is not None:
                self.last_logged_lr = current_lr
        
        # Add step count and progress
        step_metrics["train/step_count"] = self.step_count
        step_metrics["train/progress_remaining"] = self.model._current_progress_remaining
        
        # Log step metrics if any
        if step_metrics:
            wandb.log(step_metrics, step=self.step_count)
        
        self.current_episode_reward += self.locals.get('rewards', [0])[-1]
        self.current_episode_length += 1
        
        # Get PSNR values and bitrate from environment info
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'y_psnr' in info:
                    self.episode_psnr_y.append(info['y_psnr'])
                if 'cb_psnr' in info:
                    self.episode_psnr_cb.append(info['cb_psnr'])
                if 'cr_psnr' in info:
                    self.episode_psnr_cr.append(info['cr_psnr'])
                if 'bitstream_size' in info:
                    self.episode_bitrates.append(info['bitstream_size'])
        
        # Check if episode is done
        if self.locals.get('dones', [False])[-1]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log episode metrics
            episode_metrics = {
                "episode/reward": self.current_episode_reward,
                "episode/length": self.current_episode_length,
                "episode/number": len(self.episode_rewards),
                "episode/mean_y_psnr": np.mean(self.episode_psnr_y) if self.episode_psnr_y else 0,
                "episode/mean_cb_psnr": np.mean(self.episode_psnr_cb) if self.episode_psnr_cb else 0,
                "episode/mean_cr_psnr": np.mean(self.episode_psnr_cr) if self.episode_psnr_cr else 0,
                "episode/mean_bitrate": np.mean(self.episode_bitrates) if self.episode_bitrates else 0,
                "episode/total_bitrate": np.sum(self.episode_bitrates) if self.episode_bitrates else 0,
            }
            
            wandb.log(episode_metrics)
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_psnr_y = []
            self.episode_psnr_cb = []
            self.episode_psnr_cr = []
            self.episode_bitrates = []
            
        return True


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
    parser.add_argument("--batch_size", type=int, default=65, help="Batch size")
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
    
    parser.add_argument(
        "--wandb",
        help="enable wandb logging, put any value here to enable",
        default=None,
    )
    
    # Optimizer parameters
    parser.add_argument(
        "--optimizer", 
        choices=["adam", "sgd", "rmsprop"], 
        default="adam", 
        help="Optimizer to use"
    )
    parser.add_argument(
        "--adam_eps", 
        type=float, 
        default=1e-5, 
        help="Adam optimizer epsilon"
    )
    parser.add_argument(
        "--adam_betas", 
        nargs=2, 
        type=float, 
        default=[0.9, 0.999], 
        help="Adam optimizer beta parameters"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay (L2 regularization)"
    )
    
    # Scheduler parameters - now using SB3 native scheduling
    parser.add_argument(
        "--scheduler", 
        choices=["none", "linear", "cosine", "exponential"], 
        default="cosine", 
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--scheduler_final_lr_ratio", 
        type=float, 
        default=0.1, 
        help="Final learning rate as ratio of initial LR (for linear/exponential)"
    )
    parser.add_argument(
        "--scheduler_cosine_min_ratio", 
        type=float, 
        default=0.0, 
        help="Minimum LR ratio for cosine annealing"
    )
    
    # Learning rate logging parameters
    parser.add_argument(
        "--lr_log_freq", 
        type=int, 
        default=50, 
        help="Log learning rate every N steps (default: 50)"
    )

    args = parser.parse_args()
    return args

    # Create trainer and run pipeline
    # trainer = VideoEncodingTrainer(args)
    # trainer.run_complete_pipeline()


if __name__ == "__main__":

    args = prase_arg()
    # Initialize wandb
    if args.wandb is not None:
        wandb.init(
            project="av1-video-encoding",
            config={
                "algorithm": args.algorithm,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "n_steps": args.n_steps,
                "lambda_rd": args.lambda_rd,
                "total_iteration": args.total_iteration,
                "video_file": args.file,
                "optimizer": args.optimizer,
                "adam_eps": args.adam_eps,
                "adam_betas": args.adam_betas,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "scheduler_final_lr_ratio": args.scheduler_final_lr_ratio,
                "scheduler_cosine_min_ratio": args.scheduler_cosine_min_ratio,
            },
            name=f"{args.algorithm}_lr{args.learning_rate}_opt{args.optimizer}_sch{args.scheduler}"
        )
        wandb_callback = WandbCallback(lr_log_freq=args.lr_log_freq)

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

    # Set up optimizer configuration
    optimizer_kwargs = {}
    if args.optimizer == "adam":
        optimizer_kwargs = {
            "eps": args.adam_eps,
            "betas": args.adam_betas,
            "weight_decay": args.weight_decay
        }
    elif args.optimizer == "sgd":
        optimizer_kwargs = {
            "weight_decay": args.weight_decay,
            "momentum": 0.9  # Default SGD momentum
        }
    elif args.optimizer == "rmsprop":
        optimizer_kwargs = {
            "weight_decay": args.weight_decay,
            "alpha": 0.99,  # Default RMSprop alpha
            "eps": 1e-8
        }
    
    # Policy kwargs for custom optimizer
    if args.optimizer == "rmsprop":
        optimizer_class = torch.optim.RMSprop
    else:
        optimizer_class = getattr(torch.optim, args.optimizer.title())
    
    policy_kwargs = {
        "optimizer_class": optimizer_class,
        "optimizer_kwargs": optimizer_kwargs,
    }
    
    # Create SB3-native learning rate schedule
    learning_rate_schedule = create_lr_schedule(
        args.scheduler, 
        args.learning_rate,
        final_lr_ratio=args.scheduler_final_lr_ratio,
        cosine_min_ratio=args.scheduler_cosine_min_ratio
    )

    model = None
    match args.algorithm.lower():
        case "ppo":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate_schedule,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.1,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.03,
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
        case "dqn":
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=learning_rate_schedule,
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
                policy_kwargs=policy_kwargs,
                verbose=1,
            )
        case other:
            raise ValueError(f"Unsupported algorithm '{other}'. Choose either 'ppo' or 'dqn'.")
        
    total_timesteps = args.total_iteration * gyn_env.num_frames
    
    # Set up callbacks
    callbacks = []
    if args.wandb:
        callbacks.append(wandb_callback)
    
    # training
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
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