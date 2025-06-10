import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from pyencoder.environment.av1_running_env import EncoderCallback
from pyencoder.environment.naive_env import Av1Env

class VideoEncodingTrainer:
    """Complete training pipeline for video encoding RL"""
    
    def __init__(self, args):
        self.args = args
        self.video_path = args.file
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create callback instance
        self.encoder_callback = EncoderCallback(args)
        
        # Environment will be created after baseline
        self.env = None
        self.model = None
        
    def run_baseline(self):
        """Step 1: Run baseline encoding to get reference performance"""
        print("=" * 60)
        print("STEP 1: Running baseline encoding")
        print("=" * 60)
        
        baseline_output = self.output_dir / "baseline_output.ivf"
        self.encoder_callback.run_baseline_encoder(str(baseline_output))
        
        print(f"Baseline encoding completed. Output saved to: {baseline_output}")
        print("Baseline stats can be analyzed for comparison with RL performance")
        
    def create_environment(self):
        """Step 2: Create RL environment"""
        print("=" * 60)
        print("STEP 2: Creating RL environment")
        print("=" * 60)
        
        # Create main environment
        self.env = Av1Env(
            video_path=self.video_path,
            encoder_callback=self.encoder_callback,
            lambda_rd=self.args.lambda_rd,
            max_frames_per_episode=self.args.max_frames
        )
        
        # Wrap with Monitor for logging
        log_path = self.output_dir / "monitor_logs"
        log_path.mkdir(exist_ok=True)
        self.env = Monitor(self.env, str(log_path))
        
        print(f"Environment created:")
        print(f"  Action space: {self.env.action_space}")
        print(f"  Observation space: {self.env.observation_space}")
        print(f"  Max frames per episode: {self.args.max_frames}")
        
    def create_model(self):
        """Step 3: Create SB3 model"""
        print("=" * 60)
        print("STEP 3: Creating SB3 model")
        print("=" * 60)
        
        if self.args.algorithm == "ppo":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.args.learning_rate,
                n_steps=self.args.n_steps,
                batch_size=self.args.batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                tensorboard_log=str(self.output_dir / "tensorboard")
            )
        elif self.args.algorithm == "dqn":
            # Note: DQN doesn't directly support MultiDiscrete action spaces
            # You might need to flatten the action space or use a wrapper
            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=self.args.learning_rate,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=self.args.batch_size,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                verbose=1,
                tensorboard_log=str(self.output_dir / "tensorboard")
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.args.algorithm}")
        
        print(f"Created {self.args.algorithm.upper()} model")
        print(f"Model parameters: {self.model.get_parameters()}")
        
    def setup_callbacks(self):
        """Step 4: Setup training callbacks"""
        print("=" * 60)
        print("STEP 4: Setting up training callbacks")
        print("=" * 60)
        
        # Evaluation callback
        eval_env = Av1Env(
            video_path=self.video_path,
            encoder_callback=self.encoder_callback,
            lambda_rd=self.args.lambda_rd,
            max_frames_per_episode=self.args.max_frames // 2  # Shorter episodes for eval
        )
        eval_env = Monitor(eval_env, str(self.output_dir / "eval_monitor"))
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.output_dir / "best_model"),
            log_path=str(self.output_dir / "eval_results"),
            eval_freq=self.args.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=2  # Quick evaluation
        )
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.args.save_freq,
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix=f"{self.args.algorithm}_video_encoding"
        )
        
        self.callbacks = [eval_callback, checkpoint_callback]
        print(f"Setup {len(self.callbacks)} callbacks")
        
    def train_model(self):
        """Step 5: Train the model"""
        print("=" * 60)
        print("STEP 5: Training the model")
        print("=" * 60)
        
        print(f"Starting training for {self.args.total_timesteps} timesteps...")
        
        try:
            self.model.learn(
                total_timesteps=self.args.total_timesteps,
                callback=self.callbacks,
                tb_log_name=f"{self.args.algorithm}_run"
            )
            
            # Save final model
            final_model_path = self.output_dir / f"final_{self.args.algorithm}_model"
            self.model.save(str(final_model_path))
            print(f"Training completed! Final model saved to: {final_model_path}")
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            # Save current model
            interrupted_model_path = self.output_dir / f"interrupted_{self.args.algorithm}_model"
            self.model.save(str(interrupted_model_path))
            print(f"Model saved to: {interrupted_model_path}")
        
    def evaluate_model(self):
        """Step 6: Evaluate trained model"""
        print("=" * 60)
        print("STEP 6: Evaluating trained model")
        print("=" * 60)
        
        # Run final evaluation
        eval_episodes = 3
        total_rewards = []
        
        for episode in range(eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            print(f"Evaluation episode {episode + 1}/{eval_episodes}")
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                if done:
                    total_rewards.append(episode_reward)
                    print(f"Episode {episode + 1} reward: {episode_reward:.4f}")
                    break
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"Evaluation completed:")
        print(f"  Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"  Episodes: {eval_episodes}")
        
        # Run RL-controlled encoding with final model
        rl_output = self.output_dir / "rl_controlled_output.ivf"
        print(f"Running final RL-controlled encoding...")
        
        # Here you would run the encoder with the trained model
        # This requires careful synchronization as implemented in the environment
        
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("Starting complete video encoding RL training pipeline")
        print(f"Video: {self.video_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Algorithm: {self.args.algorithm}")
        
        try:
            # Step 1: Baseline
            self.run_baseline()
            
            # Step 2: Environment
            self.create_environment()
            
            # Step 3: Model
            self.create_model()
            
            # Step 4: Callbacks
            self.setup_callbacks()
            
            # Step 5: Training
            self.train_model()
            
            # Step 6: Evaluation
            self.evaluate_model()
            
            print("=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise
        finally:
            # Cleanup
            if self.env:
                self.env.close()

def main():
    parser = argparse.ArgumentParser(description="Train RL agent for video encoding")
    
    # Video and output
    parser.add_argument("--file", required=True, help="Input video file")
    parser.add_argument("--output_dir", default="./rl_training_output", 
                       help="Output directory for models and logs")
    
    # RL parameters
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo",
                       help="RL algorithm to use")
    parser.add_argument("--total_timesteps", type=int, default=50000,
                       help="Total training timesteps")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--n_steps", type=int, default=2048,
                       help="Number of steps per update (PPO only)")
    
    # Environment parameters
    parser.add_argument("--lambda_rd", type=float, default=0.1,
                       help="Rate-distortion lambda")
    parser.add_argument("--max_frames", type=int, default=100,
                       help="Maximum frames per episode")
    
    # Training parameters
    parser.add_argument("--eval_freq", type=int, default=5000,
                       help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=10000,
                       help="Model save frequency")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.file).exists():
        raise FileNotFoundError(f"Video file not found: {args.file}")
    
    # Create trainer and run pipeline
    trainer = VideoEncodingTrainer(args)
    trainer.run_complete_pipeline()

if __name__ == "__main__":
    main()