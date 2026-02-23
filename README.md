<div align="center">

# SURPASS: Autonomous Surgery

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This repo contains code for automating surgical tasks at Johns Hopkins University using the da-Vinci Reaserch Kit (dVRK).

<div align="center">

# Installation

<div align="left">

## Prerequisites

### DinoV3 
We use DinoV3 in this repository. You will need to request access to this via Huggingface. Do this [here](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m). 

## Installation Choices


There are a few different ways to install depending on where you will be using the repository. The basic process is to set up a conda environment in which you can install dependencies. Below are the 3 different ways to set up environments depending on where you will use the repo:

1. **Using and HPC Cluster with Singularity** - If your cluster does not allow docker containers or conda environments.
2. **Using a docker container** - If you have access to Docker (Strongly Recommended).
3. **Using a basic conda environment** - This is possible, but brittle on many systems (Using Docker is better).

## HPC Cluster with Singularity (DSAI Cluster)

<details>
<summary>1. Procure an interactive node</summary>

```bash
srun -p a100 --gres=gpu:1 --pty bash
```

This should pop you into a GPU node with an A100. You can check if you have access to a GPU with `nvidia-smi`.

</details>

<br>

**NOTE: The below assumes you are logged into an HPC cluster and are logged into a GPU node within an interactive shell.**

<details>
<summary>2. Create singularity container</summary>

Copy the contents of the [surpass.def](./surpass.def) file into a file of the same name in the location where you wish to build your singularity container.

Build the container with: 

```bash
apptainer build --sandbox surpass.sandbox surpass.def
```
</details>

<details>
<summary>3. Enter the singularity container in an interactive terminal</summary>

Export the location of your singularity sandbox directory.
```bash
export SANDBOX="<path/to/surpass.sandbox>"
```

To enter into the sandbox directory interactively, you can run:

**NOTE** - **!!!VERY IMPORTANT!!!**: Ensure that your workspace directory is within the `/home/<user>/scratchmunbera1` directory. If it is anywhere else, you will run out of storage space.

```bash
apptainer shell --nv --writable \
    --bind <path/to/workspace_dir>:/home/<user>/<workspace_dir_name> \
    HF_TOKEN=<your_huggingface_token_here> # for using dinov3
    $SANDBOX
```

</details>

## Docker Container

<details>
<summary>1. Build the docker image.</summary>

In the base directory of the repo, run:

```bash
docker build -t autonomous_surgery:latest .
```
</details>

<details>
<summary>2. Set up docker-compose.yaml file</summary>

There are various environment variables that need to be set uniquely for each user in the `docker-compose.yaml` file. Set these.

</details>

<details>
<summary>3. Start the Docker container.</summary>

You can run the container in headless mode and then connect via VS Code: `Dev Containers: Attach to Running Container`:

```bash
chmod +x run_container.sh
./run_container.sh
```

Alternatively, you can immediately enter the container in an interactive terminal with:

```bash
chmod +x run_container.sh
./run_container.sh -i
```

</details>

## Conda Environment

If you want to just create a conda environment without containerization, continue to the next section on installing dependencies. Do this at your own risk, as a Docker container is generally going to be more robust.

#

<div align="center">

# Dependencies

<div align="left">

You can create a conda environment and install all dependencies with:

```bash
conda create -n autonomous_surgery python=3.11
conda activate autonomous_surgery
pip install -r requirements.txt
pip install -e .
conda install -c conda-forge ffmpeg=7
```


<div align="center">

# Dataset

<div align="left">

This repository uses the Lerobot format for training. Ensure your dataset is formatted appropriately in your huggingface dataset directory. To find set this directory, you can use the environment variable `HF_HOME`.

<div align="center">

# Training

<div align="left">

To train a model, you can run:

```bash
python -m autonomous_surgery.tools.train_representation_policy
```

The configs for training can be found in `autonomous_surgery/config`.


