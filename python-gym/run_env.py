import argparse

from pyencoder.environment.av1_env import Av1Env



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)
    args = parser.parse_args()
    
    env = Av1Env(args.file)
    
    env
    
