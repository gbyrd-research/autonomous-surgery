<div align="center">

# SURPASS: Autonomous Surgery

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This repo contains code for automating surgical tasks at Johns Hopkins University using the da-Vinci Reaserch Kit (dVRK).

<div align="center">

# Installation

<div align="left">

## Setting Up HPC Cluster Environment

**NOTE: The below assumes you are logged into an HPC cluster and are logged into a GPU node within an interactive shell.**

### Create singularity container

Copy the contents of the [install](./install) directory to the location where you wish to build your singularity container.

Build the container: 

```bash
apptainer build --sandbox surpass.sandbox surpass.def
```

### Install Dependencies
<details>
<summary>1. Create a conda environment and install necessary dependencies</summary>

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n autonomous_surgery python=3.11
conda activate autonomous_surgery

# install main torch dependencies
pip install torch==2.10.0 torchvision==0.25.0
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
</details>

<details>
<summary>2. Clone this repo and install the git submodules.</summary>

```bash
git clone git@github.com:gbyrd-research/autonomous-surgery.git
cd autonomous-surgery
git submodule update --init --recursive
```
</details>

<details>
<summary>3. Install dependencies of models</summary>

```bash
# R3M（A Universal Visual Representation for Robot Manipulation）
pip install git+https://github.com/facebookresearch/r3m.git --no-deps

# CLIP (Contrastive Language-Image Pre-Training)
pip install --no-build-isolation git+https://github.com/openai/CLIP.git --no-deps

# VC1（Visual Cortex）
cd third_party/eai-vc/vc_models
pip install -e . --no-deps
cd ../../..

# SPA（3D SPatial-Awareness Enables Effective Embodied Representation）
cd third_party/SPA 
pip install --no-build-isolation . --no-deps
cd ../..
```
</details>

<details>
<summary>4. Install dependencies of simulation environments</summary>

```bash
# Metaworld
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld

# RLBench
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
cd third_party/RLBench
pip install -e .
cd ../..
```

</details>

<details>
<summary>5. Install the lift3d package.</summary>

```bash
pip install -e .

# PointNext
cd lift3d/models/point_next
cd openpoints/cpp/pointnet2_batch
pip install --no-build-isolation .
# you may need to downgrade numpy before the below installation
pip uninstall -y numpy
pip install numpy==1.23.5
# additionally, you will need to downgrade setuptools before these installations
pip uninstall -y setuptools
pip install "setuptools<60"
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

<details>
<summary>6. Quick gymnasium fix</summary>

```bash
pip install -U gymnasium
```
</details>


# Running the code

## Debugging
<details>
<summary>
To use the Visual Studio Code's debugger, the following must be done.</summary>

1. Ensure your VS Code is running **ON THE COMPUTE NODE YOU WISH TO RUN THE CODE ON.**
2. Inside your singularity container and inside your conda environment, initialize `debugpy` on the file you wish to debug. Use the `--wait-for-client` flag to initialize the debugger and wait for a client to connect. For example:

```bash
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m lift3d.tools.train_representation_policy
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

## Generate simulation dataset

To perform a sanity check training on simulation data, you can create a simulation dataset with the following command:

```bash
python -m lift3d.scripts.gen_data_metaworld
```

This will generate data for many different tasks. You can stop the generation after the first task is completely generated with ~30 episodes. Or, you can leave the script running to generate massive amounts of data. For a sanity check, one task dataset will be sufficient.

## Run Training with Representation-ACT

To train on your generated data, run the following:

```bash
python -m lift3d.tools.train_representation_policy
```


