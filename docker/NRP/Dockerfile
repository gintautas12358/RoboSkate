FROM hbpneurorobotics/nrp-core:unity

RUN sudo apt-get -y update \
    && sudo apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    xvfb \
    libxrandr2 \
    && sudo apt-get -y install python3-pip \
    && sudo apt-get clean\
    && sudo apt-get -y install python \
    && sudo apt-get install -y python-concurrent.futures


COPY requirements.txt /tmp/


RUN \
    pip3 install --upgrade protobuf && \
    pip3 uninstall -y stable-baselines3 && \
    pip3 install Pillow==7.2.0 && \
    pip3 install -r /tmp/requirements.txt && \
    pip3 install git+https://github.com/eleurent/highway-env && \
    rm -rf $HOME/.cache/pip


CMD /bin/bash
