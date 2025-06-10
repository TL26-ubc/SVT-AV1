import argparse

import pyencoder
from pyencoder.environment.av1_running_env import Av1RunningEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)
    parser.add_argument("--output", help="Output video file", required=False, default="Output/output.ivf")

    args = parser.parse_args()
    print(args)
    
    the_callback = Av1RunningEnv(args)
    the_callback.run_SVT_AV1_encoder(first_round=True)
    the_callback.run_SVT_AV1_encoder(args.output, first_round=False)
    
