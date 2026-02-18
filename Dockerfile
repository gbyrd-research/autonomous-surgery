FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Basic utilities (NO colcon here)
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    build-essential \
    git \
    wget \
    locales \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Locale setup
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 + colcon AFTER repo exists
RUN apt-get update && apt-get install -y \
    ros-galactic-desktop \
    python3-rosdep \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Auto-source ROS
RUN echo "source /opt/ros/galactic/setup.bash" >> /root/.bashrc

# Install conda
ENV CONDA_DIR=/opt/conda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Make conda available in all bash sessions
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc

SHELL ["/bin/bash", "-c"]
CMD ["bash"]
