"""
Multiprosessing learning environment for DummyBall with A2C
"""

import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import subprocess
from multiprocessing import Process
import threading
import os
import time
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from scripts.python.CallBack import SaveOnBestTrainingRewardCallback, TensorboardCallback
from scripts.python.DummyBall.DummyBallGymEnv_image import DummyBallEnv

n_parallel_env = 1
agent_name = "A2C_agent_canny_image_cnn_policy_150"
render_image_width = 150
render_image_height = 150
base_Port = 50051
log_interval = 1

if __name__ == '__main__':

    # --------------------------------------------------------------------------------
    # ------------------ SubprocVecEnv ----------------------------------------------
    # --------------------------------------------------------------------------------
    # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv
    # https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/3_multiprocessing.ipynb#scrollTo=kcYpsA8ExB9T

    def DummyBallEnv_thread(port):
        # TODO: implement for Windows and Linux
        # For Mac
        #os.system("./games/RoboSkate.app/Contents/MacOS/RoboSkate", "-p " + port)
        # For Linux
        os.system("./games/linux/NRPDummyBallGame/NRPDummyBallGame.x86_64 -p " + port)


    def make_env(rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():

            port = str(base_Port + rank)
            threading.Thread(target=DummyBallEnv_thread, args=(port,)).start()

            # TODO: Use command reply instead of 15 sec delay
            time.sleep(15)
            print("started")
            env = DummyBallEnv(render_image_width, render_image_height, port)


            # Important: use a different seed for each environment
            env.seed(seed + rank)
            return env

        return _init


    if n_parallel_env == 1:
        # if there is only one process, there is no need to use multiprocessing
        train_env = DummyVecEnv([make_env(0)])
    else:
        # Here we use the "spawn" method for launching the processes, more information is available in the doc
        train_env = SubprocVecEnv([make_env(i) for i in range(n_parallel_env)], start_method='spawn')
        #train_env.seeds()  # Sets the random seeds for all environments.


    # --------------------------------------------------------------------------------
    # ------------------ training ----------------------------------------------------
    # --------------------------------------------------------------------------------

    train_env.reset()

    for i in range(5):

        # define save dir
        log_dir = "./scripts/python/agent_models/" + agent_name

        # Create callbacks
        tensor_callback = TensorboardCallback(n_parallel_env)

        # CnnPolicy SB3 policy for images as observations
        model = A2C("CnnPolicy", train_env, tensorboard_log="./scripts/python/tensorboard/" + agent_name)
        #model = A2C.load(log_dir, train_env)
        model.learn(total_timesteps=20000, log_interval=log_interval, callback=[tensor_callback])
        model.save(log_dir + str(i))

    model = A2C.load(log_dir, train_env)
    obs = train_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info_list = train_env.step(action)

        print(reward)

    # Clean up the environment’s resources.
    train_env.close()

