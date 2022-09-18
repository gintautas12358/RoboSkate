"""
This is supposed to be a skeleton for a SAC agent playing RoboSkate.

"sac_roboskate.zip" is the current model. Since until now the state space is not implemented properly, the model is useless.
"""


from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import subprocess
from multiprocessing import Process
import threading
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
from scripts.python.CallBack import SaveOnBestTrainingRewardCallback, TensorboardCallback
from scripts.python.DummyBall.DummyBallGymEnv import DummyBallEnv

agent_name = "SAC_agent"
render_image_width = 50
render_image_height = 50
base_Port = 50051
n_parallel_env = 1
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
    # TODO: use Callback for save agent model

    # define save dir
    log_dir = "./scripts/python/agent_models/" + agent_name

    # Create callbacks
    tensor_callback = TensorboardCallback(n_parallel_env)
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

    model = SAC('MlpPolicy', train_env, tensorboard_log="./scripts/python/tensorboard/" + agent_name)
    model.learn(total_timesteps=500, log_interval=log_interval, callback=[tensor_callback])
    model.save(log_dir)

    model = SAC.load(log_dir, train_env)
    obs = train_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info_list = train_env.step(action)
        info_dict = info_list[0]
        print(obs)
        print(reward)
        if (info_dict["time"]) >= 20:
            print("time up")
            break

    train_env.close() # Clean up the environment’s resources.





