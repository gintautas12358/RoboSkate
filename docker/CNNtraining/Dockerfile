#FROM hbpneurorobotics/nrp-core:unity
FROM ubuntu:latest

# Install apt-get packages
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt-get -y clean

# Install torch CPU
#RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# Install torch GPU
#RUN pip3 install torch torchvision torchaudio
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Install all pip packages
#RUN pip install torch
#RUN pip install tensorboard



COPY entrypoint.sh /tmp/
RUN chmod +x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]


CMD /bin/bash