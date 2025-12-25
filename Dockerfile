FROM nvcr.io/nvidia/cuda:13.0.2-runtime-ubuntu22.04

# for avoidance of 'tzdata' configuring
ENV DEBIAN_FRONTEND=noninteractive
# for GUI
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
# Change the apt-server
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.riken.jp/Linux/ubuntu/%g" /etc/apt/sources.list

RUN apt-get update && apt-get install -y\
    python3-pip python3-dev wget git\
    x11-apps python3-pyqt5\
    libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf msttcorefonts\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY ./requirements.txt .
RUN pip install -r requirements.txt
