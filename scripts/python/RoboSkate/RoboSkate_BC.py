"""
RoboSkate Behavioral cloning.

this script is no longer up to date!
It was used to learn a policy from the expert data. This works very well.
"""

import numpy as np
import pickle
import tempfile
import pathlib
from scripts.python.RoboSkate.Environments.Env_joints_and_direction_error import RoboSkateEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import stable_baselines3

from stable_baselines3.common.vec_env import SubprocVecEnv
import threading
import time
from sys import platform
from imitation.algorithms import bc

import os
from imitation.data import rollout
from imitation.util import logger

from tensorboard import program

# --------------------------------------------------------------------------------
# ------------------ Variables  --------------------------------------------------
# --------------------------------------------------------------------------------



use_camera = False
render_image_width = 200
render_image_height = 200
base_Port = 50051
agent_name = "BC_directionError_level1"
run_on = "local"

if platform == "darwin":
    Training = False
    n_parallel_env = 1
    training_epochs = 40
    trajectories_to_load = 3
    AutostartRoboSkate = False
elif platform == "linux" or platform == "linux2":
    Training = True
    n_parallel_env = 1
    training_epochs = 200000
    run_on = "server"
    trajectories_to_load = 100
    AutostartRoboSkate = True


max_episode_length = 800






if __name__ == '__main__':

    def Tensorboard_thread():
        if (run_on == 'server'):
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--bind_all', '--port', '8080', '--logdir', './scripts/python/RoboSkateIL/tensorboard/' + agent_name + '/'])
            url = tb.launch()
            print("Access TensorBoard: http://10.195.6.144:8080 (use LRZ VPN)")
        else:
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', './scripts/python/RoboSkateIL/tensorboard/' + agent_name + '/'])
            url = tb.launch()
            print("Access TensorBoard: http://localhost:6006")

    # --------------------------------------------------------------------------------
    # ------------------ SubprocVecEnv -----------------------------------------------
    # --------------------------------------------------------------------------------

    def RoboSkate_thread(port):
        if platform == "darwin":
            os.system("./games/RoboSkate.app/Contents/MacOS/RoboSkate -p " + port)
        elif platform == "linux" or platform == "linux2":
            print("Linux")
            os.system("games/RoboSkate/roboskate.x86_64 -nographics -batchmode -p " + port)
        elif platform == "win32":
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
            if AutostartRoboSkate:
                threading.Thread(target=RoboSkate_thread, args=(port,)).start()

                # TODO: Use command reply instead of 15 sec delay
                time.sleep(15)
                print("environoment " + str(rank) + " started.")
            else:
                print("RoboSkate needs to be started manual.")
            env = RoboSkateEnv(max_episode_length, use_camera, render_image_width, render_image_height, port)

            # Important: use a different seed for each environment
            env.seed(seed + rank + np.random.randint(100))

            return env

        return _init



    # --------------------------------------------------------------------------------
    # ------------------ immitation lerning ------------------------------------------
    # --------------------------------------------------------------------------------

    print("create environment.")
    venv = SubprocVecEnv([make_env(i) for i in range(n_parallel_env)], start_method='spawn')

    if Training:
        threading.Thread(target=Tensorboard_thread).start()

        # iterate over files in directory
        directory = "./scripts/python/RoboSkateIL/RoboSkateExpertData/directionError/"
        for filename in os.listdir(directory):
            with open(directory + "/" + filename, "rb") as f:

                try:
                    trajectories
                except NameError:
                    trajectories = pickle.load(f)
                else:
                    trajectories.extend(pickle.load(f))
            # stop early if enough trajectories a loaded
            if len(trajectories) >= trajectories_to_load:
                break

        print(str(len(trajectories)) + " trajectories were loaded.")


        # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
        # This is a more general dataclass containing unordered
        # (observation, actions, next_observation) transitions.
        transitions = rollout.flatten_trajectories(trajectories)




        tempdir = tempfile.TemporaryDirectory(prefix=agent_name)
        tempdir_path = pathlib.Path('./scripts/python/RoboSkateIL/tensorboard/' + agent_name + '/')
        print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

        # Train BC on expert data.
        # BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
        # dictionaries containing observations and actions.
        logger.configure(tempdir_path)
        '''
        bc_trainer = bc.BC(venv.observation_space,
                           venv.action_space,
                           expert_data=transitions,
                           ent_weight=1e-4,
                           l2_weight=1e-2,
                           policy_class=ActorCriticPolicy,
                           policy_kwargs=dict(
                               model_class=A2C))
        '''
        bc_trainer = bc.BC(venv.observation_space,
                           venv.action_space,
                           expert_data=transitions,
                           policy_class=ActorCriticPolicy

        )

        bc_trainer.train(n_epochs=training_epochs)

        bc_trainer.save_policy("./scripts/python/RoboSkateIL/agent_models/BC/" + agent_name)

    if platform == "darwin":
        new_policy = bc.reconstruct_policy("./scripts/python/RoboSkateIL/agent_models/BC/" + agent_name)
        print(new_policy)
    
        print("Run Agent")
        obs = venv.reset()
        while True:
            action, _states = new_policy.predict(obs, deterministic=False)
            obs, reward, done, info_list = venv.step(action)
            # print(obs)
            info_dict = info_list[0]
            if (info_dict["time"]) >= 100:
                print("time is up.")
                break

    venv.close()
    print("environoment closed.")





