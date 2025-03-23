from argparse import ArgumentParser
from pathlib import Path
from subprocess import run
from sys import argv

from .global_variables import copy_baseline
'''
Sample usage:

python new_package/tl26/main.py -e ~/tl26/SVT-AV1/Bin/Debug/SvtAv1EncApp -i ~/tmp/playground/akiyo_qcif.y4m -b ~/tmp/playground/output.ivf --rc 1 --tbr 1000
'''
parser = ArgumentParser(
                    prog='TL26_RL_SVT-AV1',
                    description='train and test RL agent with SVT-AV1')

parser.add_argument('-i', '--input', required=True, type=Path,
                    help='input video file')
parser.add_argument('-b', '--output', required=True, type=Path,
                    help='output video file')
parser.add_argument('--rc', type=str, default='1',
                    help='rate control mode')
parser.add_argument('--tbr', type=str, default='1000',
                    help='target bitrate')
parser.add_argument('-e', '--executable', type=Path, default='SvtAv1EncApp',
                    help='path to SVT-AV1 executable')
parser.add_argument('--train_epochs', type=int, default=1000)
# TODO: Add support for other flags
# parser.add_argument('-f', '--flags', type=str, default='',
#                     help='SVT-AV1 other flags')

args = parser.parse_args(argv[1:])

# So this is the first round of training, use the original encoding parameters, collect baseline
# TODO: Need to somehow tell the agent to not train
run([args.executable, '-i', args.input, '-b', args.output, '--rc', args.rc, '--tbr', args.tbr])

copy_baseline() # Read only

# Now train the agent
for epoch in range(args.train_epochs):
    # TODO: Need to somehow tell the agent to train
    run([args.executable, '-i', args.input, '-b', args.output, '--rc', args.rc, '--tbr', args.tbr])