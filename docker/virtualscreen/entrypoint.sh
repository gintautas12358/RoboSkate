#!/bin/bash
# This script is the entrypoint for the Docker image.

cd /home/ubuntu/roboskate/
# make RoboSkate executable
chmod -R 755 games/RoboSkate/roboskate.x86_64

cd /home/ubuntu/roboskate/gym/
pip3 install -e .

cd /home/ubuntu/roboskate/stable-baselines3/
pip3 install -e .


# Start XServer
echo 'Start XServer'
nohup Xorg vt1 :1 > xserver.log &

cd /home/ubuntu/roboskate/
# start RoboSkate on Display 1
echo 'Start RoboSkate3 on Port 50051'
DISPLAY=:1 nohup games/RoboSkate3/roboskate.x86_64  > roboskate3.log &

# wait 5 sec
sleep 10

#echo 'Start Training'

cd /home/ubuntu/roboskate/rl-baselines3-zoo/

#python3 train.py --algo sac --env RoboSkateSegmentation-v0 --tensorboard-log logs/tensorboard --n-timesteps 1000000 --seed 1 --env-kwargs port:50051 headlessMode:False AutostartRoboSkate:False --save-replay-buffer

set -e
exec "$@"
