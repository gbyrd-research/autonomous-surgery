# ------------------------------------------------------------
# Base: CUDA + Ubuntu 20.04 (required for ROS Noetic)
# ------------------------------------------------------------
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ------------------------------------------------------------
# Base utilities
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    gnupg2 \
    lsb-release \
    build-essential \
    ca-certificates \
    sudo \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install ROS Noetic (system Python 3.8)
# ------------------------------------------------------------
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros1.list

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

# ------------------------------------------------------------
# Optional: Install Miniforge (isolated from ROS)
# This does NOT interfere with system Python
# ------------------------------------------------------------
ENV CONDA_DIR=/opt/conda

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Create ML environment (separate from ROS)
RUN conda create -n ml python=3.10 -y && \
    conda clean -afy

# ------------------------------------------------------------
# Install autonomous_surgery dependencies into system Python
# (ROS Noetic Python runtime)
# ------------------------------------------------------------
WORKDIR /tmp/autonomous_surgery_build

COPY requirements.txt pyproject.toml README.md ./
COPY autonomous_surgery ./autonomous_surgery

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir -e .

# ------------------------------------------------------------
# Setup ROS environment
# ------------------------------------------------------------
SHELL ["/bin/bash", "-c"]

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /ros1_ws/devel/setup.bash ]; then source /ros1_ws/devel/setup.bash; fi" >> /root/.bashrc

# ------------------------------------------------------------
# Create Catkin Workspace
# ------------------------------------------------------------
RUN mkdir -p /ros1_ws/src
WORKDIR /ros1_ws

RUN source /opt/ros/noetic/setup.bash && \
    catkin init

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

RUN /opt/conda/bin/python3 -m pip install \
catkin_pkg \
rospkg \
empy \
PyYAML \
setuptools \
numpy==1.26.4 \
opencv-python

RUN conda install -y libstdcxx-ng

ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# ------------------------------------------------------------
# Default entry
# ------------------------------------------------------------
CMD ["bash"]
