#!/bin/bash
# This script is the entrypoint for the Docker image.

pip uninstall -y gym
cd /home/ubuntu/roboskate/gym
pip install -e .

pip uninstall -y stable-baselines3
cd /home/ubuntu/roboskate/stable-baselines3
pip install -e .

#cd /home/ubuntu/roboskate/rl-baselines3-zoo
#pip install -r requirements.txt


cd /home/ubuntu/roboskate

# make RoboSkate executable
chmod -R 755 games/RoboSkate/roboskate.x86_64

# open ssh connection to other server only possible for training without images
# ssh -4 -fNT -i /root/.ssh/RoboSkateCCLRZprivateKey \
# -L 50010:127.0.0.1:50010 \
# -L 50011:127.0.0.1:50011 \
# -L 50012:127.0.0.1:50012 \
# -L 50013:127.0.0.1:50013 \
# -L 50014:127.0.0.1:50014 \
# -L 50015:127.0.0.1:50015 \
# -L 50016:127.0.0.1:50016 \
# -L 50017:127.0.0.1:50017 \
# -L 50018:127.0.0.1:50018 \
# -L 50019:127.0.0.1:50019 \
# -l ubuntu 10.195.6.144

# Run training
# python3 -m scripts.python.RoboSkateIL.PPO > PPO_output_log_file.log
cd /home/ubuntu/roboskate/rl-baselines3-zoo
python3 train.py --algo sac --env RoboSkateNummerical-v0 --tensorboard-log logs/tensorboard --n-timesteps 1000000 --env-kwargs headlessMode:False

