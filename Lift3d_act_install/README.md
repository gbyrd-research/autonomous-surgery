# SURPASS Compute Environment

We will be using [singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) containers to create development environments for our research code during the SURPASS project.

## Installation

A `base.def` is provided in this directory. Running the following command will create a dev environment with the following dependencies:

- Ubuntu 22.04
- Cuda 12.1
- Python 3.11
- Pytorch 2.4.0
- Torchvision 0.19.0
- Pytorch3d (a prebuilt .whl is provided in this repository also since compiling from source is difficult on JHU's HPC cluster)

### Run the following to build the singularity container

```bash
apptainer build --sandbox surpass.sandbox surpass.def
```

And enter the container in the CLI with

```bash
apptainer shell --nv --writable surpass.sandbox
```

### Install Dependencies
<details>
<summary>Create a conda environment and install necessary dependencies</summary>

```bash
cd Lift3d_act
source /opt/conda/etc/profile.d/conda.sh
conda create -n lift3d_act python=3.11
conda activate lift3d_act

# install main torch dependencies
pip install torch==2.4.0 torchvision==0.19.0
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
</details>

<details>
<summary>Initialize the git submodules</summary>

```bash
git submodule update --init --recursive
```
</details>

<details>
<summary>Install dependencies of models</summary>

```bash
# R3M（A Universal Visual Representation for Robot Manipulation）
pip install git+https://github.com/facebookresearch/r3m.git --no-deps

# CLIP (Contrastive Language-Image Pre-Training)
pip install git+https://github.com/openai/CLIP.git --no-deps

# VC1（Visual Cortex）
cd third_party/eai-vc/vc_models
pip install -e .  --no-deps
cd ../../..

# SPA（3D SPatial-Awareness Enables Effective Embodied Representation）
cd third_party/SPA 
pip install -e . --no-deps
cd ../..
```
</details>

<details>
<summary>Set environment variables</summary>

```bash
# WandB
export WANDB_API_KEY=<wandb_api_key>
export WANDB_USER_EMAIL=<wandb_email>
export WANDB_USERNAME=<wandb_username>

# CoppeliaSim & PyRep & RLBench
export COPPELIASIM_ROOT=${HOME}/Programs/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:99  # required on server, remove it on workstation
```

</details>

<details>
<summary>Install dependencies of simulation environments</summary>

```bash
# Metaworld
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld

# RLBench
wget --no-check-certificates https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
cd third_party/RLBench
pip install -e .
cd ../..
```

</details>

<details>
<summary>Install the Lift3D package.</summary>

```bash
pip install -e .

# PointNext
cd lift3d/models/point_next
cd openpoints/cpp/pointnet2_batch
pip install --no-build-isolation .
cd ../subsampling
pip install --no-build-isolation .
cd ../pointops
pip install --no-build-isolation .
cd ../chamfer_dist
pip install --no-build-isolation .
cd ../emd
pip install --no-build-isolation .
cd ../../../../../..
```
</details>


## Debugging
<details>
<summary>
To use the Visual Studio Code's debugger, the following must be done.</summary>

1. Ensure your VS Code is running **ON THE COMPUTE NODE YOU WISH TO RUN THE CODE ON.**
2. Inside your singularity container and inside your conda environment, initialize `debugpy` on the file you wish to debug. Use the `--wait-for-client` flag to initialize the debugger and wait for a client to connect. For example:

```bash
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client eval.py
```
3. Add something similar to the below in your `.vscode/launch.json` file:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach to Singularity Python",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    # path to the workspace folder OUTSIDE your singularity container
                    "localRoot": "${workspaceFolder}",

                    # path to the workspace folder INSIDE your singularity container
                    "remoteRoot": "/home/gbyrd/SURPASS"
                }
            ]
        }
    ]
}
```
4. Click on the debug icon in VS Code and press the play button after selecting `Attach to Singularity Python`
</details>