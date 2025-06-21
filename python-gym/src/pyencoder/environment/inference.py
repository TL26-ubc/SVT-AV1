import argparse
from pyencoder.environment.av1_running_env import Av1RunningEnv
from pyencoder.environment.naive_env import Av1GymEnv
from stable_baselines3 import DQN, PPO
from pathlib import Path



def parse_arg():
    parser = argparse.ArgumentParser(description="Infernece an  RL agent for video encoding")
    
    # Video and output
    parser.add_argument(
        "--file", required=True, help="Input video file"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output file for the encoded video"
    )
    parser.add_argument(
        "--output_dir", default="logs/", help="Output directory for models and logs"
    )
    
    # model parameters
    parser.add_argument(
        "--model_path", required=True, help="Path to the trained model file"
    )
    
    return parser.parse_args()

    

if __name__ == "__main__":
    args = parse_arg()
    
    output_dir = Path(args.output_dir)
    env = Av1GymEnv(        
        video_path=args.file,
        output_dir=output_dir,
        inference=True,  # Set inference mode
        inference_path=args.output_file,  # Path to save inference results
        lambda_rd=0.1,
    )
    
    if "ppo" in args.model_path:
        model = PPO.load(args.model_path, env=env)
        args.algorithm = "ppo"
    elif "dqn" in args.model_path:
        model = DQN.load(args.model_path, env=env)
        args.algorithm = "dqn"
    else:
        raise ValueError("Unsupported model type. Please provide a PPO or DQN model.")
    
    video_length = env.num_frames
    
    print(f"Running inference for {video_length} frames using {args.algorithm.upper()} model.")
    
    # inference loop
    model.learn(
        total_timesteps=video_length - 1,
    )