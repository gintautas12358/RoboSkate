# G1 RoboSkate

* [Report](./documentation/Report/RoboSkate_report.pdf)


* [KickOff project presentation](./documentation/Presentations/1_kickoff_project_presentation.pdf)
* [Midterm project presentation](./documentation/Presentations/2_midterm_project_presentation.pdf)
* [Final project presentation](./documentation/Presentations/3_final_project_presentation.pdf)


In most subfolders there are README files that explain further things. Under documentation you can also find the final report, important terminal commands and more information about the RoboSkate interface.

## OpenAI Gym 
This repository contains important scripts, trained models, datasets and documentation to use RoboSkate as an OpenAI Gym Environment and to train it easily with the tools of rl-baselines3-zoo. The OpenAI Gym Environment is at this time (07/15/2021) still in a submodule hosted on GitHub.com.

Under trained_agents you can find trained agents (+Tensorboard logs) as well as trained models for a VAE to extract features from the image data of RoboSkate which can then be fed into the training.

Under scripts are tools like a remote control for the RoboSkate agent, something to train the VAE as well as a script to collect image data. There are also older scripts that are not yet adapted to the current environoments but are interesting as soon as the topics become interesting again, e.g. Behavior cloning.

submodule rl-baselines3-zoo is a fork of the original and contains training parameters for RoboSkate.

submodule gym is a fork of the original and contains Open AI RoboSkate environments.

submodule stable-baselines3 is a fork of the original and contains a multi input policy for the associated Gym Environment.

Under expert_data, for example, labeled images can be found.


## NRP

This repository contains documentation and code for training artificial agents to play Roboskate through the [Neurorobotics platform](https://neurorobotics.net/) (NRP). Up until now, the NRP only supported Gazebo as a simulator option. For this project, we will be using the new version of the NRP which introduces the possibility of different simulators to be used by specifying an interface that can be connected to the NRP. In our case, this will be a Unity-developed game [RoboSkate](https://store.steampowered.com/app/1404530/RoboSkate/), for which we have worked with the game developer to define the interface and connect the game with the NRP. Therefore, alongside the main task of training RL agents to play the game, it is important to test how this can be done in a workflow with the new version of the NRP and to provide some statistics about this during the Masterpraktikum.

## Quick start RoboSkate

First, in the _games_ folder there are links for the games to be downloaded (contains Linux and Windows options).
There are 2 games: NRPDummyBallGame and RoboSkate.

- The NRPDummyBallGame is a test game you can use for debugging, testing the pipeline and troubleshooting.
- The RoboSkate is the game to use for the final task.

Both games actually contain a [gRPC](https://grpc.io/)-based server that starts once the game is loaded and waits for requests on port 50051. Once the server is started, client programs can connect to it and communicate. The API is explained in _documentation\roboskate_api_v3.pdf_ and example code that uses the api through python can be found in scripts\python\client.py.

The scripts/python_nrp contains the example of how to connect to the game through the new version of the Neurorobotics platform, which is available from the predefined docker image hbpneurorobotics/nrp-core:unity. For this, you first need to install [docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on your machine. This approach should be the final workflow and will only work on Linux so far (tested on Ubuntu 18.04 and Ubuntu 20.04, while steps 1 and 2 described below you can also test on Windows in the early stage of the Masterpraktikum). Once you have installed the prerequisites, you can use the following commands to start the game:

`xhost + local:`

`docker run --runtime=nvidia -e DISPLAY -v $PWD:/home/nrpuser/nrp-core/examples/roboskate:rw -v /tmp/.X11-unix:/tmp/.X11-unix:rw --privileged -it --name nrp_game hbpneurorobotics/nrp-core:unity bash -c "source /home/nrpuser/.bashrc;cd home/nrpuser/nrp-core/examples/roboskate/;NRPSimulation -c scripts/python_nrp/simulation_config.json"`

Note: _abs/path/to/the/root/folder/of/the/repo/here_ should be modified to the repository root folder on your system and the games need to be downloaded in the _games_ folder (check the README in the _games_ folder). The `xhost + local:` should be run before the docker command to allow local connections to xhost because the game is started from within the container. You can modify the _scripts/python_nrp/simulation_config.json_ to start either the NRPDummyBallGame or RoboSkate by providing the relative path to the executable. Make sure you have run `chmod +x` on the game executables in order to be able to start the games. 

Things to test step by step:

1. Only try to start the game as a normal executable, if everything works fine: 
- for NRPDummyBallGame you should get a green-blue screen after the unity splash screen and the game loading
- for RoboSkate after the unity and the game splash screen the scene should be loaded and you should see the robo-skateboard
- Note: it is normal to get a "force quit" option when you start the games as the UI might get unresposive while they load, just wait a little longer
2. If a game is started successfully, use the scripts/python/client.py to connect to the game and provide actions and receive observations from the game. If communication is successful, you should get the observations in your python console and also images from the game camera should be saved in the _images_ folder
3. If 1 and 2 worked, and you have installed the prerequisites mentioned above, you can use the docker command from above which creates a container based on the hbpneurorobotics/nrp-core:unity image. First, make sure that you allow writing in the _images_ folder from within the docker container, by executing `chmod 777 images` as the current user outside of the docker container. The last part of the above command i.e. `_NRPSimulation -c scripts/python_nrp/simulation_config.json_` starts the NRP within the container which in turn starts the game. If this is successful, first the game should start loading and then it should start running, while also you should see output in the console and images being saved in the _images_ folder.

Note: The RoboSkate game might be quite demanding and only work if you have a powerful enough GPU on the Linux machine, on WIndows it should work fine in general. We will proceed to see how to optimize this based on the results you get when you try the above steps with RoboSkate. But all of the above steps should easily work for the NRPDummyBallGame. 

## Troubleshooting:

**Problem:** _No protocol specified_ message in console when you try to start the game throug the docker:

- you probaly forgot to run `xhost + local:` before starting the container
- maybe the nvidia-docker is not installed or not working correctly

**Problem** _PermissionError: [Errno 13] Permission denied: images/received-image-0.jpg_

- you probably forgot to run `chmod 777 images` such that the docker user can wirte files in the folder created by you as user
- you probably created the images as one user (e.g., as the docker user) but now you are running without docker trying to overwrite the existing image with your current user

## Quick start Open AI Gym
If this repository is cloned, it must be ensured that the submodules are also available. The submodules are located on GitHub.com
In addition, the Branch RoboSkate must be checkedout so that the relevant code parts are available.

e.g. `/gym/gym/envs/RoboSkate/` must be available


https://github.com/Finn-Sueberkrueb/rl-baselines3-zoo.git

https://github.com/Finn-Sueberkrueb/stable-baselines3.git

https://github.com/Finn-Sueberkrueb/gym.git

gym and stable baselines 3 must be installed before it can be used.

`cd ./gym`\
`pip install -e .`

`cd ./stable-baselines3`\
`pip install -e .`

### Training
Training can be started directly from the rl-baselines3-zoo folder. (RoboSkate game should be running) 

`python train.py --algo sac --env RoboSkateNumerical-v0`

algo: reinforcement algorithm. More detail in rl-baselines3-zoo/README.dm \
env: gym environment. All environments and more detail are in gym/gym/env/RoboSkate

Trained agent is saved in rl-baselines3-zoo/logs/#algo#/ (replace "#algo#" with RL algorithm you used)


### Running agents

Running a pretrained agent can be done with the command directly from the rl-baselines3-zoo folder:

simple:\
`python enjoy.py --algo sac --env RoboSkateNumerical-v0`

advance:\
`python enjoy.py --algo sac --env RoboSkateNumerical-v0 --folder logs/ --load-best --env-kwargs headlessMode:False random_start_level:False startLevel:0 --exp-id 1`

folder: More detail in rl-baselines3-zoo/README.dm \
load-best: More detail in rl-baselines3-zoo/README.dm \
env-kwargs: More detail in rl-baselines3-zoo/README.dm and in the used environment gym/gym/env/RoboSkate \
exp-id: More detail in rl-baselines3-zoo/README.dm

### Running best agents

Best models are saved in trained_models/. They can be run directly from  the rl-baselines3-zoo folder:

`python enjoy.py --algo sac --env RoboSkateNumerical-v0 --folder ../trained_models/RoboSkate --load-best --env-kwargs headlessMode:False random_start_level:False startLevel:0 --exp-id 1`

List of available agents:\
RoboSkateNumerical-v0 with indexes 1 to 4 \
RoboSkateSegmentation-v0 with indexes 1 to 3

More details in trained_models\README.dm


## Docker image

The docker images shiped with Stable Baseline3 (SB3) _https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html_ and rl-baselines3-zoo (currently tested: enjoy.py, train.py) (record_video.py has problems).

To build Dockerfile:

`docker build -t nrp ./docker/NRP` 

Here "-t nrp" is for tagging the image.

To open a cmd in the container:

`docker run --name rl -it  nrp`

The folder named rl_zoo contains _https://github.com/DLR-RM/rl-baselines3-zoo.git_, whereas nrp_core - NRP simulation.

















