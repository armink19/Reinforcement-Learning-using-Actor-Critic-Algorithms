# Reinforcement Learning using Actor Critic Algorithms

## Installation Guide for RoboGym with MuJoCo

This guide provides step-by-step instructions for installing and setting up RoboGym with MuJoCo on the UL HPC platform.

### Prerequisites

Before starting the installation process, ensure you have the following:

- Python 3.8.6
- Virtual Environment (venv)
- Internet connectivity

### Setup Python Environment

```bash
module load lang/Python/3.8.6-GCCcore-10.2.0
python3 -m venv rlenv
source ./rlenv/bin/activate
```

### Install MuJoCo Binary

```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
mv mujoco210 ~/.mujoco
```

### Install Python MuJoCo Bindings

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin/
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip3 install -e .
```

### Test MuJoCo Import

Before testing, install the required dependency:

```bash
pip install cython==0.29.36
```

Test the MuJoCo import:

```bash
python -c "import mujoco_py"
```

If it fails due to missing `GL/osmesa.h`, proceed to the next step.

### Install Diverse Dependencies

```bash
module load vis/PyOpenGL
pip install patchelf
```

### Install RoboGym

```bash
git clone https://github.com/openai/robogym.git
cd robogym
```

Edit `setup.py` by replacing the line `"mujoco-py==2.0.2.13" with "mujoco-py==2.1.2.14"`:



Install RoboGym:

```bash
pip install -e .
```

### Launch RoboGym Unit Test

```bash
pytest ./robogym/tests/test_robot_env.py
```

This completes the installation and setup process for RoboGym with MuJoCo on your system. If you encounter any issues during installation, refer to the official documentation or seek assistance from the respective repositories.
