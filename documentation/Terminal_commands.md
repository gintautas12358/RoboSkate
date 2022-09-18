# Important terminal commands
## Connect to Server
TensorBoard\
`http://10.195.6.248:8080`\
`http://10.195.6.248:8081`

CC LRZ login\
`ssh -i ~/.ssh/RoboSkateCCLRZprivateKey -l ubuntu 10.195.6.248`

Chair GPU login\
`ssh -i /~/.ssh/RoboSkateGPUprivateKey -l cmlrss2021-g1 131.159.60.36`

show GPU usage\
`$ watch -d -n 0.5 nvidia-smi`

processes\
`$ htop`

## Docker commands

Build Docker image\
`$ docker build -t roboskate:finn ./docker/RoboSkate/`\
`$ docker build -t roboskategpu2:finn ./docker/virtualscreen/`

launch Docker container\
`$  docker run -it --gpus all --rm --name RoboLevel1 --privileged --mount type=bind,source="$(pwd)",target=/home/ubuntu/roboskate roboskategpu2:finn`\
`$  docker run -it --gpus all --rm --name roboskatecnn --mount type=bind,source="$(pwd)",target=/home/ubuntu/roboskate roboskatecnn:finn`\
`$  docker run -it --rm --name givemebackmyownership --mount type=bind,source="$(pwd)",target=/home/ubuntu/roboskate givemebackmyownership:finn`

Show Docker log\
`docker logs roboskatecnn`

Open docker shell\
`$ docker exec -it roboskate /bin/bash`\
Detach the container by pressing ctrl+p and ctrl+q one after another.

List\
`$ docker container ls`\
`$ docker image list`

remove\
`$ docker rmi image_name_or_id`


## Run scripts at Server
restart server\
`$ sudo shutdown -r now`

Start script in background and log to file.\
`$ nohup python3 -m scripts.python.RoboSkate.RoboSkateSB3_A2C > output_log_file.log &`

## VM setup (obsolete)
Install Anaconda: https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-debian-10 \
`$ pip install tensorboard`\
`$ pip install gym`\
`$ pip install stable-baselines3`


## Local commands
start tensorboard\
`$ tensorboard --logdir ./scripts/python/RoboSkate/tensorboard/<agent_name>/`\
`$ tensorboard --logdir ./rl-baselines3-zoo/logs/tensorboard/`

Use remote RoboSkate\
`ssh -fNT -i /home/cmlrss2021-g1/.ssh/RoboSkateCCLRZprivateKey \
-L 50010:127.0.0.1:50010 \
-L 50011:127.0.0.1:50011 \
-L 50012:127.0.0.1:50012 \
-L 50013:127.0.0.1:50013 \
-L 50014:127.0.0.1:50014 \
-L 50015:127.0.0.1:50015 \
-L 50016:127.0.0.1:50016 \
-L 50017:127.0.0.1:50017 \
-L 50018:127.0.0.1:50018 \
-L 50019:127.0.0.1:50019 \
-l ubuntu 10.195.6.248`


## Start Docker with Remote RoboSkate
1. Start all RoboSkate instances manually on the LRZ server\
`nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50010 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50011 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50012 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50013 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50014 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50015 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50016 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50017 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50018 &
nohup RoboSkate/games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p 50019 &`

2. run container on Chair server with the corresponding number of environments


## Connect TensorBoard from Chair Server to LRZ server

`chmod 400 ~/.ssh/RoboSkateGPUprivateKey`

1. login to LRZ Server
2. `$ mkdir /home/ubuntu/ChairGPUtensorboard`
3. `$ sshfs cmlrss2021-g1@131.159.60.36:/home/cmlrss2021-g1/G1_RoboSkate/rl-baselines3-zoo/logs/tensorboard/ /home/ubuntu/ChairGPUtensorboard/ -o IdentityFile=/home/ubuntu/.ssh/RoboSkateGPUprivateKey`
4. `$ nohup tensorboard --logdir /home/ubuntu/ChairGPUtensorboard/ --port=8081 --bind_all > TensorBoardRemote.out &`
5. (unmount agein: `$ umount /home/ubuntu/ChairGPUtensorboard/`)


## Start traing with RL Baselines3 ZOO
Use rl-baselines3-zoo as root folder.

enjoy best agent local\
`python enjoy.py --algo sac --env RoboSkateNumerical-v0 --folder logs/ --load-best --env-kwargs headlessMode:False random_start_level:False startLevel:0 --exp-id 4`

`python enjoy.py --algo sac --env RoboSkateSegmentation-v0 --folder logs/ --load-best --env-kwargs headlessMode:False random_start_level:False startLevel:0 show_image_reconstruction:True --exp-id 1 --n-timesteps 1000`

Train Local (for testing)\
`python train.py --algo sac --env RoboSkateMultiInputPolicy-v0 --env-kwargs AutostartRoboSkate:False`



### Train on Server \

RoboSkateNumerical-v0 \
`nohup python train.py --algo sac --env RoboSkateNumerical-v0 --tensorboard-log logs/tensorboard -i ./logs/sac/RoboSkateNumerical-v0_1/best_model.zip --n-timesteps 6000000 --save-freq 50000 --env-kwargs random_start_level:False startLevel:0 startport:50051 headlessMode:True --save-replay-buffer > RoboSkateReduced_Level1.out &`

RoboSkateMultiInputPolicy-v0 \
`python3 train.py --algo sac --env RoboSkateMultiInputPolicy-v0 --tensorboard-log logs/tensorboard -i ./logs/sac/RoboSkateMultiInputPolicy-v0_3/best_model.zip --n-timesteps 6000000 --save-freq 50000 --save-replay-buffer`

RoboSkateSegmentation-v0 \
`python3 train.py --algo sac --env RoboSkateSegmentation-v0 --tensorboard-log logs/tensorboard  -i ./logs/sac/RoboSkateSegmentation-v0_level1_wrongCheckpoint/best_model.zip --n-timesteps 6000000 --save-freq 50000 --env-kwargs random_start_level:False startLevel:0 headlessMode:False --save-replay-buffer`



Start TensorBoard at LRZ Server\
`nohup tensorboard --bind_all --port 8080 --logdir /home/ubuntu/G1_RoboSkate/rl-baselines3-zoo/logs/tensorboard > TensorBoardLocal.out &`
