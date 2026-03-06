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
# Setup ROS environment
# ------------------------------------------------------------
SHELL ["/bin/bash", "-c"]

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# ------------------------------------------------------------
# Create Catkin Workspace
# ------------------------------------------------------------
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

RUN source /opt/ros/noetic/setup.bash && \
    catkin init

# ------------------------------------------------------------
# Default entry
# ------------------------------------------------------------
CMD ["bash"]