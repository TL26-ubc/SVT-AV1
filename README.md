# TL26 SVT-AV1 with RL

This repository contains the code for the RL-based optimization of the SVT-AV1 encoder. The code is based on the [SVT-AV1](https://gitlab.com/AOMediaCodec/SVT-AV1) encoder.

The original readme which contains the documentation for the SVT-AV1 encoder can be found [here](Original_SVTAV1_Readme.md).

## System Prerequisites

Before setting up the Python environment, ensure you have the following system-level dependencies installed. You should use your operating system's package manager (e.g., apt on Debian/Ubuntu, yum on Fedora, Homebrew on macOS, or Chocolatey/winget on Windows) to install them.

1.  **Basic Development Tools**:
    -   `cmake`
    -   `ninja`
    -   `nasm`
    -   `gcc` and `g++` (or `clang` and `clang++` on macOS)
    -   `python3-dev` (or `python3.x-dev` for your specific Python version)
    -   `libgl1`
```
sudo apt update && sudo apt install -y cmake ninja-build nasm build-essential
sudo apt install python3.12-dev
sudo apt install libgl1
```

1.  **Python** (Recommended 3.9-3.11):
    -   Install `python` (e.g., `python@3.11`).
    *Note: Make sure this Python version is used for your virtual environment.*

2.  **FFmpeg** (for video processing):
    -   Install `ffmpeg`.

3.  **NASM** (Assembler, required by SVT-AV1):
    -   Install `nasm`.

## Environment Setup

**Papers and Blogs**
- [Netflix Blog 2020](https://netflixtechblog.com/svt-av1-an-open-source-av1-encoder-and-decoder-ad295d9b5ca2)
- [SPIE 2020](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11510/1151021/The-SVT-AV1-encoder--overview-features-and-speed-quality/10.1117/12.2569270.full)
- [SPIE 2021](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11842/118420T/Towards-much-better-SVT-AV1-quality-cycles-tradeoffs-for-VOD/10.1117/12.2595598.full)
- [SVT-AV1 - Tech Blog 2022](https://networkbuilders.intel.com/blog/svt-av1-enables-highly-efficient-large-scale-video-on-demand-vod-services)
- [SPIE 2022](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12226/122260S/Enhancing-SVT-AV1-with-LCEVC-to-improve-quality-cycles-trade/10.1117/12.2633882.full)
- [Adaptive Steaming Common Test Conditions](https://aomedia.org/docs/SIWG-D001o.pdf)
- [ICIP 2023](https://arxiv.org/abs/2307.05208)
- [SPIE 2024](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13137/131370W/Benchmarking-hardware-and-software-encoder-quality-and-performance/10.1117/12.3031754.full)
- [ICIP 2024](https://aomedia.org/docs/Software_Implementation_Working_Group_Update_ICIP2024.pdf)

1. Create a new Python virtual environment:
   ```bash
   python -m venv venv
   ```

**Technical Appendices**
- [Adaptive Prediction Structure Appendix](Docs/Appendix-Adaptive-Prediction-Structure.md)
- [Altref and Overlay Pictures Appendix](Docs/Appendix-Alt-Refs.md)
- [CDEF Appendix](Docs/Appendix-CDEF.md)
- [CfL Appendix](Docs/Appendix-CfL.md)
- [Compliant Subpel Interpolation Filter Search Appendix](Docs/Appendix-Compliant-Subpel-Interpolation-Filter-Search.md)
- [Compound Mode Prediction Appendix](Docs/Appendix-Compound-Mode-Prediction.md)
- [Deblocking Loop Filter (LF) Appendix](Docs/Appendix-DLF.md)
- [Film Grain Synthesis](Docs/Appendix-Film-Grain-Synthesis.md)
- [Global Motion Appendix](Docs/Appendix-Global-Motion.md)
- [Intra Block Copy Appendix](Docs/Appendix-Intra-Block-Copy.md)
- [Local Warped Motion appendix](Docs/Appendix-Local-Warped-Motion.md)
- [Mode Decision Appendix](Docs/Appendix-Mode-Decision.md)
- [Motion Estimation Appendix](Docs/Appendix-Open-Loop-Motion-Estimation.md)
- [Overlapped Block Motion Compensation Appendix](Docs/Appendix-Overlapped-Block-Motion-Compensation.md)
- [Palette Prediction Appendix](Docs/Appendix-Palette-Prediction.md)
- [Rate Control Appendix](Docs/Appendix-Rate-Control.md)
- [Recursive Intra Appendix](Docs/Appendix-Recursive-Intra.md)
- [Restoration Filter Appendix](Docs/Appendix-Restoration-Filter.md)
- [SQ Weight Appendix](Docs/Appendix-SQ-Weight.md)
- [Super-resolution Appendix](Docs/Appendix-Super-Resolution.md)
- [Temporal Dependency Model](Docs/Appendix-TPL.md)
- [Transform Search Appendix](Docs/Appendix-TX-Search.md)
- [Reference Scaling Appendix](Docs/Appendix-Reference-Scaling.md)
- [Variance Boost Appendix](Docs/Appendix-Variance-Boost.md)

**How Can I Contribute?**
- [SVT-AV1 Contribution Guide](Docs/Contribute.md)
