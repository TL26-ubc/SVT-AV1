# TL26 SVT-AV1 with RL

This repository contains the code for the RL-based optimization of the SVT-AV1 encoder. The code is based on the [SVT-AV1](https://gitlab.com/AOMediaCodec/SVT-AV1) encoder.

The original readme which contains the documentation for the SVT-AV1 encoder can be found [here](Original_SVTAV1_Readme.md).

## System Prerequisites

Before setting up the Python environment, ensure you have the following system-level dependencies installed. You should use your operating system's package manager (e.g., apt on Debian/Ubuntu, yum on Fedora, Homebrew on macOS, or Chocolatey/winget on Windows) to install them.

1.  **Basic Development Tools**:
    -   `cmake`
    -   `ninja`
    -   `git`

2.  **Python** (Recommended 3.9-3.11):
    -   Install `python` (e.g., `python@3.11`).
    *Note: Make sure this Python version is used for your virtual environment.*

3.  **FFmpeg** (for video processing):
    -   Install `ffmpeg`.

4.  **NASM** (Assembler, required by SVT-AV1):
    -   Install `nasm`.

## Environment Setup

To set up the development environment, follow these steps:

1. Create a new Python virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Unix or MacOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the `python-gym` package:
   This package contains the custom Reinforcement Learning environment.
   ```bash
   cd python-gym
   pip install -e .
   cd ..
   ```

The `requirements.txt` file includes all necessary packages for:
- Machine Learning (PyTorch, Stable-Baselines3, Gymnasium)
- Data Processing (NumPy, Pandas, OpenCV)
- Visualization (Matplotlib, Seaborn)
- Development Tools (pybind11)