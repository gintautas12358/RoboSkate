"""
RoboSkate
this script is no longer up to date!
"""
import numpy as np
from scripts.python.RoboSkate.Environments.Env_images_and_direction_error import RoboSkateEnv
from gym.envs.RoboSkate.RoboSkatePosVelImage import  RoboSkatePosVelImage
from scripts.python.RoboSkate.hardcoded_agents import HardcodedAgents
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import pickle
import imageio



import threading
import os
import time

from sys import platform

from imitation.data import rollout, types


# --------------------------------------------------------------------------------
# ------------------ Variables  --------------------------------------------------
# --------------------------------------------------------------------------------



collect_image_data = True
render_image_width = 100
render_image_height = 100
base_Port = 50051

n_parallel_env = 1
max_episode_length = 1000
episodes_per_file = 1
max_file_number = 1






if __name__ == '__main__':
    # --------------------------------------------------------------------------------
    # ------------------ SubprocVecEnv -----------------------------------------------
    # --------------------------------------------------------------------------------

    def RoboSkate_thread(port):
        if platform == "darwin":
            os.system("./games/RoboSkate.app/Contents/MacOS/RoboSkate -p " + port)
        elif platform == "linux" or platform == "linux2":
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
            threading.Thread(target=RoboSkate_thread, args=(port,)).start()

            # TODO: Use command reply instead of 15 sec delay
            time.sleep(15)
            print("environoment " + str(rank) + " started.")
            #env = RoboSkateEnv(max_episode_length, collect_image_data, render_image_width, render_image_height, port)
            env = RoboSkatePosVelImage(1000, 50051, True, False)


            # Important: use a different seed for each environment
            env.seed(seed + rank + np.random.randint(100))

            return env

        return _init

    # --------------------------------------------------------------------------------
    # ------------------ immitation lerning ------------------------------------------
    # --------------------------------------------------------------------------------


    print("create environment.")
    env = make_vec_env(make_env(0), n_envs=1, vec_env_cls=SubprocVecEnv)



    # collect image and direction error
    image_data = []
    with open("/RoboSkateExpertData/directionError/image_data.pkl", "rb") as fp:  # Pickling
        image_data = pickle.load(fp)

    offset = len(image_data)


    # accumulator for incomplete trajectories
    trajectories_accum = rollout.TrajectoryAccumulator()

    # obs[0] since we use SubprocVecEnv with 1 environoment
    obs = env.reset()[0]
    agent = HardcodedAgents(obs, True)

    # Hardcoded agent
    # Collect rollout tuples.
    trajectories = []



    trajectories_accum.add_step(dict(obs=obs), 0)

    step = 0
    runs = 0

    print("start hardcoded agent")
    while True:

        # Get the new actions from the agent
        # actions = agent.joints3_XYPos_Cases_Level1()
        #actions = np.array([agent.joints3_DirectionError1()])
        actions = np.array([agent.Tele_joints3_3Vel_DirectionError1()])

        # Simulate the environment for one step
        obs, reward, done, info = env.step(actions)

        # Only possible for one environoment
        agent.obs = obs[0]
        agent.reward = reward[0]
        agent.done = done[0]
        agent.info = info[0]
        agent.step = step


        print("step: " + str(step))
        step += 1

        # append new trajectorie
        new_trajs = trajectories_accum.add_steps_and_auto_finish(actions, obs, reward, done, info)
        trajectories.extend(new_trajs)

        if collect_image_data:
            # append new image with ground truth
            image = agent.info['image']
            imageio.imwrite("./scripts/python/RoboSkateIL/RoboSkateExpertData/directionError/images/RoboSkate-" + str(step + offset) + ".png", image)
            direction_error = agent.obs[7]
            print(direction_error)
            image_data.append(["RoboSkate-" + str(step + offset) + ".png", direction_error])


        # Collect a amount of runs
        if agent.done:
            runs += 1
            print("runs done: " + str(runs))
        if runs >= episodes_per_file:
            break


    print("End data collection")

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    np.random.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        print("Sanity check")
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + env.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + env.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    print("done Sanity checks")

    # check if a file with the name already exists
    filenr = 0
    while True:
        if os.path.isfile("./scripts/python/RoboSkateIL/RoboSkateExpertData/directionError/" + str(filenr) + "trajectorie.pkl"):
            filenr += 1
        else:
            types.save("./scripts/python/RoboSkateIL/RoboSkateExpertData/directionError/" + str(filenr) + "trajectorie.pkl", trajectories)
            break

    if collect_image_data:
        # store image data list
        with open("/RoboSkateExpertData/directionError/image_data.pkl", "wb") as fp:  # Pickling
            pickle.dump(image_data, fp)

    print("saved files!")




    # Clean up the environment’s resources.
    env.close()
    print("Program finished.")

