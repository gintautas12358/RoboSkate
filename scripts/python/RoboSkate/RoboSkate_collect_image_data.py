"""
RoboSkate collect image data
"""
import numpy as np
from gym.envs.RoboSkate.RoboSkateNumerical import RoboSkateNumerical
from scripts.python.RoboSkate.hardcoded_agents import HardcodedAgents
import imageio
from PIL import Image as im


from sys import platform

number_of_images_to_collect = 400
image_counter = 0
step = 0
useTeleoperation = False
level = 1

if __name__ == '__main__':

    print("create environment.")
    env = RoboSkateNumerical(max_episode_length=10000,
                    startport=50051,
                    rank=0,
                    headlessMode=False,
                    AutostartRoboSkate=True,
                    startLevel=level,
                    cameraWidth=1000,
                    cameraHeight=300)


    obs = env.reset()
    # one class for all different types of hardcoded agents
    if useTeleoperation:
        agent = HardcodedAgents(obs, True)
    else:
        agent = HardcodedAgents(obs, False)

    print("start hardcoded agent")
    while number_of_images_to_collect > image_counter:

        agent.step = step
        step += 1

        if useTeleoperation:
            actions, stop, level = agent.Teleoperation()
            env.startLevel = level
        else:
            actions = agent.checkpoint_follower()
            stop = False
            env.startLevel = level


        if stop:
            env.reset()
            agent.obs = env.reset()
        else:
            # Simulate the environment for one step
            agent.obs, agent.reward, agent.done, agent.info = env.step(actions)

        if agent.done:
            env.reset()
            agent.obs = env.reset()

        if step%5 == 0:
            # append new image with ground truth
            image = agent.info['image'].transpose([1, 2, 0])
            imageio.imwrite("./scripts/python/RoboSkate/RoboSkateExpertData/Segmentation/images/RoboSkate-" + str(image_counter) + ".jpg", image)
            image_counter += 1



    print("End data collection")

    # Clean up the environment’s resources.
    env.close()
