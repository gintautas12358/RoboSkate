"""
Multiprosessing learning environment for DummyBall with A2C
"""

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env, is_wrapped
import subprocess
from multiprocessing import Process
import threading
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
from scripts.python.CallBack import SaveOnBestTrainingRewardCallback, TensorboardCallback
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from scripts.python.DummyBall.DummyBallGymEnv import DummyBallEnv
from scripts.python.DummyBall.DummyBallGymEnv_image import DummyBallEnvImage
from tensorboard import program

from sys import platform

n_parallel_env = 2
agent_name = "PPO_agent"
render_image_width = 50
render_image_height = 50
base_Port = 50051
log_interval = 32
checkpoint_save_freq = 1000
total_training_timesteps = 10000000

n_epochs = 2 # Number of epoch when optimizing the surrogate loss
n_steps = 8 # The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs)
batch_size = 16 # use batch_size that is a multiple of n_steps * n_envs

#best_agent_eval_freq = 1000
#best_agent_n_eval_episodes = 2

RLalgo = PPO
Training = True
policy = "CnnPolicy" # "MlpPolicy" or CnnPolicy
device_for_training = 'cpu'
use_images = True # "MlpPolicy" or CnnPolicy
run_on_server = False

if __name__ == '__main__':
    # --------------------------------------------------------------------------------
    # ------------------ Tensorboard  ------------------------------------------------
    # --------------------------------------------------------------------------------
    def Tensorboard_thread():
        if run_on_server:
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--bind_all', '--port', '8080', '--logdir',
                               './scripts/python/DummyBall/tensorboard/' + agent_name + '/'])
            url = tb.launch()
            print("Access TensorBoard: http://10.195.6.144:8080 (use LRZ VPN)")
        else:
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', './scripts/python/DummyBall/tensorboard/' + agent_name + '/'])
            url = tb.launch()
            print("Access TensorBoard: http://localhost:6006")


    # callback for plotting additional values in tensorboard.
    class TensorboardCallback(BaseCallback):

        def __init__(self, verbose=2):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # SubprocVecEnv
            #self.logger.record('State/JointAngle/0', self.locals.get("new_obs")[0][0] * 180)
            self.logger.record('reward/mean', np.mean(self.locals.get("rewards")[0]))

            return True


    # --------------------------------------------------------------------------------
    # ------------------ SubprocVecEnv ----------------------------------------------
    # --------------------------------------------------------------------------------
    # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv
    # https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/3_multiprocessing.ipynb#scrollTo=kcYpsA8ExB9T

    def DummyBallEnv_thread(port):
        # choose Platform if not on server
        if platform == "darwin":
            # For Mac
            os.system("./games/NRPDummyBallGame.app/Contents/MacOS/NRPDummyBallGame -p " + port)
        elif platform == "linux" or platform == "linux2":
            os.system("./games/linux/NRPDummyBallGame/NRPDummyBallGame.x86_64 -p " + port)
        elif platform == "win32":
            # TODO: implement for Windows
            os.system("./games/RoboSkate/RoboSkate.exe -p " + port)



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
            if use_images:
                env = DummyBallEnvImage(render_image_width, render_image_height, port)
            else:
                env = DummyBallEnv(render_image_width, render_image_height, port)


            # Important: use a different seed for each environment
            env.seed(seed + rank)
            return env

        return _init

    train_env = SubprocVecEnv([make_env(i+1) for i in range(n_parallel_env)], start_method='spawn')
    #eval_env = make_vec_env(make_env(0), n_envs=1, vec_env_cls=SubprocVecEnv)

    # --------------------------------------------------------------------------------
    # ------------------ training ----------------------------------------------------
    # --------------------------------------------------------------------------------

    train_env.reset()
    # TODO: use Callback for save agent model

    # check if a model with the name already exists
    if os.path.isfile("./scripts/python/DummyBall/agent_models/" + agent_name + ".zip"):
        # load previously created model trained model
        model = RLalgo.load("./scripts/python/DummyBall/agent_models/" + agent_name,
                            train_env)
        print("Found previously created model.")

    elif os.path.isfile("./scripts/python/DummyBall/agent_models/best_model.zip"):
        # load previously created model trained model
        model = RLalgo.load("./scripts/python/DummyBall/agent_models/best_model",
                            train_env)
        print("Found previously saved Best model.")

    else:
        if Training:
            # create new model
            model = RLalgo(policy,
                           train_env,
                           device=device_for_training,
                           verbose=1,
                           n_steps = n_steps,
                           n_epochs = n_epochs,
                           batch_size=batch_size,
                           tensorboard_log = ('./scripts/python/DummyBall/tensorboard/' + agent_name))

            print("New model created.")
        else:
            print("No model found!")
            train_env.close()
            quit()



    if Training:
        # Start tensorboard
        threading.Thread(target=Tensorboard_thread).start()


        # Create callbacks
        checkpoint_callback = CheckpointCallback(save_freq=checkpoint_save_freq,
                                             save_path='./scripts/python/DummyBall/agent_models/checkpoint/',
                                             name_prefix=agent_name)


        #tensor_callback = TensorboardCallback(n_parallel_env)

        '''
        eval_callback = EvalCallback(eval_env, best_model_save_path=("./scripts/python/DummyBall/agent_models/"),
                                     log_path=("./scripts/python/DummyBall/agent_models/"),
                                     eval_freq=best_agent_eval_freq, n_eval_episodes=best_agent_n_eval_episodes,
                                     deterministic=False, render=False)
        '''

        print("Start training.")
        model.learn(total_timesteps=total_training_timesteps,
                    log_interval=log_interval,
                    callback=[TensorboardCallback(), checkpoint_callback])

        model.save("./scripts/python/DummyBall/agent_models/" + agent_name)
        print("Finished training.")


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

