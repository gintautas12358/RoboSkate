# Docker containers

## CNN training
This Docker can be used for training CNNs on the GPUs.
It dose not use the RoboSkate game. Only image data from a folder.


## give me back my ownership
This container is just a workaround to assign the ownership of files written by some dockercontainer back to the linux user.\
Probably there is a simpler solution for this...


## NRP
TODO: Gintautas Palinauskas add description in Docker container.\
(? Can be used to run RoboSkate in combination with NRP. There are some problems with the synchronization between environment and model.)


## RoboSkate virtualscreen
Dockerfile to run RoboSkate with graphics\
- Image depends on nvidia/vulkan https://hub.docker.com/r/nvidia/vulkan `ubuntu18.04`\
- xorg.conf is added manually because there is no access to `nvidia-xconfig`.\


## RoboSkate Numerical
Everything that is important to use and train RoboSkate in headless mode.