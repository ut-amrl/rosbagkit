FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV USERNAME=ut-amrl

# Install ROS Noetic and necessary tools
RUN apt-get update && apt-get install -y \
    lsb-release \
    gnupg2 \
    curl \
    build-essential \
    cmake \
    git \
    x11-apps \
    xauth \
    && curl -sSL http://repo.ros2.org/repos.key | apt-key add - \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    ros-noetic-ros-numpy \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
  
# Initialize rosdep
RUN rosdep init && rosdep update

# Install Miniforge
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3 && \
    rm Miniforge3-Linux-x86_64.sh

# Add mamba to PATH
ENV PATH=/opt/miniforge3/bin:$PATH

# Create a new mamba environment
COPY environment.yaml /tmp/environment.yaml
RUN mamba env create -f /tmp/environment.yaml

# Create a new user with sudo privileges
RUN useradd -ms /bin/bash $USERNAME && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME

# Set up ROS and Conda environment for the new user
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "source /opt/miniforge3/bin/activate dataset-tools" >> /home/$USERNAME/.bashrc

# Copy the updated .bashrc to root
RUN cp /home/$USERNAME/.bashrc /root/.bashrc

# Change to the new user
USER $USERNAME
WORKDIR /home/$USERNAME

# Set the entrypoint to bash to allow interactive use
ENTRYPOINT ["/bin/bash"]