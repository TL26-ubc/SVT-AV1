# how to build and run TL26 SVT-AV1 with RL

## Prerequisites if on Ubuntu

The OS dependecies are:
- cmake
- ninja
- git
- python (recommended 3.9-3.11)
- ffmpeg (for video processing)
- nasm (assembler, required by SVT-AV1)
- gcc (compiler)

Run the following command to install on ubuntu:
```bash
sudo apt-get install cmake ninja-build git python3 python3-venv ffmpeg nasm build-essential
```

Pybind11 is required to build the RL-based SVT-AV1. It will be installed in the python virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install pybind11
```

## Clone the repository

Find an empty directory and clone the repository:

```bash
git clone https://github.com/TL26-ubc/SVT-AV1.git
# checkout the branch where we are developing the RL-based SVT-AV1
git checkout tom_mod

cd SVT-AV1/python-gym
# install the python dependencies
pip install -r requirements.txt

# build the RL-based SVT-AV1
pip install -e .
```
The above should build the RL-based SVT-AV1 and install the python package `svt_av1_gym` in the virtual environment.

Then find another empty directory to clone the training repository:

```bash
git clone https://github.com/TL26-ubc/av1env-training.git
git checkout tom_mod
```

`av1env-training/src/train_refactor.py` is the main training script.

## Download an example video
We used an example video for our experient, to download it, goto where you want to store the video and run:
```bash
wget https://media.xiph.org/video/derf/y4m/football_cif.y4m
```

## Run the training
Now you can run the training script, for example:
```bash
cd av1env-training
source ../SVT-AV1/venv/bin/activate
python src/train_refactor.py --video_path ../football_cif.y4m --output_dir ./output --total_iteration 100 --wandb True
```
This will run the training for 100 iterations, and save the model and logs in the `output` directory. You can change the parameters as needed. Make sure you change the paths to the video and output directory accordingly.

The above will start wandb where you can monitor the training process. First time you run it, it will ask you to login to wandb. You can create a free account if you don't have one. Then copy the API key to the terminal to login.