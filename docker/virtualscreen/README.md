# Dockerfile to run RoboSkate with graphics

- The Server has 2x GeForce GTX 1080
- Image depends on nvidia/vulkan https://hub.docker.com/r/nvidia/vulkan `ubuntu18.04`
- xorg.conf is added manually because there is no access to `nvidia-xconfig`.
- good help: https://stackoverflow.com/questions/63483222/docker-xserver-for-nvidia-opengl-application-without-x-in-host
- Unity Log file: `nano /root/.config/unity3d/Matas\ Sakalauskas/RoboSkate/Player.log`
- actually `xserver-xorg-video-nvidia-390` should be used but this does not seem to work therefore `xserver-xorg-video-nvidia-440` is used











## How to run Unity on Amazon Cloud or without Monitor
Also good but useless without root access\
https://towardsdatascience.com/how-to-run-unity-on-amazon-cloud-or-without-monitor-3c10ce022639