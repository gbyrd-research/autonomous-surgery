<div align="center">

# SURPASS: Autonomous Surgery

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

This repo contains code for automating surgical tasks at Johns Hopkins University using the da-Vinci Reaserch Kit (dVRK).

<div align="center">

# Installation

<div align="left">

## Setting Up HPC Cluster Environment

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

**NOTE - !!!VERY IMPORTANT!!! : Ensure that your workspace directory is within the `/home/<user>/scratchmunbera1` directory. If it is anywhere else, you will run out of storage space.

```bash
apptainer shell --nv --writable \
    --bind <path/to/workspace_dir>:/home/<user>/<workspace_dir_name> \
    $SANDBOX
```

Enter the sandbox as above and then proceed to install the dependencies below.

</details>

<details>
<summary>4. Configure ssh agent if not already running</summary>

```bash
ssh-add -l
```

If you get: Could not open a connection to your authentication agent then:

```bash
# Turn on Agent
eval "$(ssh-agent -s)"

# Add key
ssh-add ~/.ssh/id_rsa

# If successful, you will see a fingerprint of your key instead of the error message.
ssh-add -l
```

</details>

## Setting Up Docker Container

<details>
<summary>1. Configure ssh agent if not already running</summary>

```bash
ssh-add -l
```

If you get: Could not open a connection to your authentication agent then:

```bash
# Turn on Agent
eval "$(ssh-agent -s)"

# Add key
ssh-add ~/.ssh/id_rsa

# If successful, you will see a fingerprint of your key instead of the error message.
ssh-add -l
```
</details>

<details>
<summary>2. Build Docker image.</summary>

```bash
docker build -t autonomous_surgery:latest .
```

</details>

<details>
<summary>3. Run docker container from autonomous_surgery Docker image.</summary>

```bash
docker compose up
```

</details>

<details>
<summary>4. Enter the container in interactive mode.</summary>

```bash
docker exec -it autonomous_surgery /bin/bash
```

</details>

Proceed to `Install Dependencies` below.


## Install Dependencies

<details>
<summary>1. Create a conda environment, immediately install Lerobot, and cd into your workspace directory.</summary>

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n autonomous_surgery python=3.11
conda activate autonomous_surgery

# lerobot must be installed before installing the downstream dependencies
# because installing lerobot after will change the pytorch installation which
# will break the downstream dependencies
pip install lerobot==0.3.3

cd </home/<user>/<workspace_dir_name>>
```
</details>

<details>
<summary>2. Clone this repo and install the git submodules.</summary>

**NOTE:** If you are running with a docker container, this step is different. See below.

```bash
git clone git@github.com:gbyrd-research/autonomous-surgery.git
cd autonomous-surgery
git submodule update --init --recursive
```

**Docker container version**

You **MUST** run the following on the host machine **outside of the docker container** before entering the container. Otherwise, you will run into challenges cloning the submodules due to user name mismatches and dubious ownership errors.

```
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
export COPPELIASIM_ROOT=${HOME}/Programs/CoppeliaSim
wget --no-check-certificate https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
cd third_party/RLBench
pip install -e .
cd ../..
```

</details>

<details>

<summary>5. Downgrade pip</summary>

```bash
pip install pip==23.3.1
```

</details>

<details>

<summary>6. Install repo as editable python package</summary>

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

<details>
<summary>7. Quick gymnasium fix</summary>

```bash
pip install -U gymnasium
```
</details>

<details>
<summary>8. Install pytorch3d</summary>

```bash
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

</details>



# Running the code

## Entering the sandbox to run the code

You will need to include various environment variables when entering the singularity sandbox to run this code. Here is my (Grayson) setup. You will need to populate some of the environment variables with your unique entries.

Reminder:

```bash
export SANDBOX="<path/to/surpass.sandbox>"
```

**NOTE: Ensure that your workspace directory is within the `/home/<user>/scratchmunbera1` directory. If it is anywhere else, you will run out of space.

```bash
apptainer shell --nv --writable \
    --bind <path/to/workspace_dir>:/home/<user>/<workspace_dir_name> \
    --env COPPELIASIM_ROOT=${HOME}/Programs/CoppeliaSim \
    --env LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH \
    --env QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT \
    --env DISPLAY=:99 \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    --env MUJOCO_GL=egl \
    --env PYOPENGL_PLATFORM=egl \
    --env HF_TOKEN=<your_huggingface_token> \
    --env HF_HOME=</home/<user>/<workspace_dir_name>/.hf> \
    --env TORCH_HOME=</home/<user>/<workspace_dir_name>/.torch> \
    $SANDBOX
```

## Debugging
<details>
<summary>
To use the Visual Studio Code's debugger, the following must be done.</summary>

1. Ensure your VS Code is running **ON THE COMPUTE NODE YOU WISH TO RUN THE CODE ON.**
2. Inside your singularity container and inside your conda environment, initialize `debugpy` on the file you wish to debug. Use the `--wait-for-client` flag to initialize the debugger and wait for a client to connect. For example:

```bash
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m lift3d.tools.train_policy_new
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

```bash
python -m lift3d.scripts.gen_data_metaworld
```