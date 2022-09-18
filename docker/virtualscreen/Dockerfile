FROM nvidia/vulkan:1.1.121-cuda-10.1-beta.1-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
        apt install -y \
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libgles2 \
        xserver-xorg-video-nvidia-440\
        nano\
        wget


RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt-get -y clean

# Copy display configuration
COPY xorg.conf /etc/X11/xorg.conf

# set Nvidia Runtime Variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV DISPLAY :1


# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN chmod -R 755 ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda

# Install all pip packages
RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN pip3 install grpcio
RUN pip3 install google
RUN pip3 install protobuf
RUN pip3 install pybullet
RUN pip3 install gym-minigrid
RUN pip3 install scikit-optimize
RUN pip3 install optuna
RUN pip3 install pytablewriter
RUN pip3 install seaborn
RUN pip3 install sb3-contrib
RUN pip3 install imageio
RUN pip3 install tensorboard
RUN pip install torchvision
RUN pip install pickle5
RUN pip install opencv-python



COPY entrypoint.sh /tmp/
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh", "/bin/bash"]




