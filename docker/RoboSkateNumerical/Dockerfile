FROM ubuntu:latest

# Install apt-get packages
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    xvfb \
    libxrandr2
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt-get -y clean

RUN apt install -y git


RUN mkdir -p /home/ubuntu/
RUN git clone https://github.com/DLR-RM/rl-baselines3-zoo.git /home/ubuntu/rl-zoo/
RUN cd /home/ubuntu

COPY requirements.txt /tmp/

ENV $PATH=/home/ubuntu/.local/bin:$PATH

RUN \
    pip3 uninstall -y stable-baselines3 && \
    pip3 install Pillow==7.2.0 && \
    pip3 install -r /tmp/requirements.txt && \
    pip3 install git+https://github.com/eleurent/highway-env && \
    rm -rf $HOME/.cache/pip




ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get -y install ssh

# For SSH connetion
COPY RoboSkateCCLRZprivateKey /root/
RUN eval `ssh-agent -s` && \
mkdir ~/.ssh && \
mv ~/RoboSkateCCLRZprivateKey ~/.ssh/ && \
echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
cat /etc/ssh/ssh_config && \
chmod go-w /root && \
chmod 700 /root/.ssh && \
chmod 600 /root/.ssh/RoboSkateCCLRZprivateKey && \
ssh-add ~/.ssh/RoboSkateCCLRZprivateKey


# Install torch CPU
# RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# Install torch GPU
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Install all pip packages

RUN pip install grpcio
RUN pip install google
RUN pip install protobuf
RUN pip install torch
RUN pip install tensorboard
#RUN pip install imitation



RUN pip install gym
#RUN pip uninstall -y stable-baselines3
RUN pip install stable-baselines3

# RUN pip install pybullet
# RUN pip install gym-minigrid
# RUN pip install scikit-optimize
# RUN pip install optuna
# RUN pip install pytablewriter
# RUN pip install seaborn
# RUN pip install pyyaml
# RUN pip install sb3-contrib
# RUN pip install box2d-py
# RUN pip install atari-py
# RUN pip install plotly
# RUN pip install opencv-python
# RUN pip install psutil


RUN pip install imageio



COPY entrypoint.sh /tmp/
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]


CMD /bin/bash
