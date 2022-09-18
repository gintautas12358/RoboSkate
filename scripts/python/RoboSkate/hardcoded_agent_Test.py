"""
simple function to test the gym environment and move the agent hardcoded
This script is used to test the hardcoded agents. It starts an environment and then connects the selected agent to the environment.

"""

from gym.envs.RoboSkate.RoboSkateNumerical import RoboSkateNumerical
from gym.envs.RoboSkate.RoboSkateSegmentation import RoboSkateSegmentation
from scripts.python.RoboSkate.hardcoded_agents import HardcodedAgents

useTeleoperation = True
max_episode_length = 4000
level = 1

# create the Gym environment. The RoboSkate game needs to be started manual
env = RoboSkateSegmentation(max_episode_length=max_episode_length,
                         startport=50051,
                         rank=0,
                         headlessMode=False,
                         AutostartRoboSkate=True,
                         startLevel=level,
                         small_checkpoint_radius=False,
                         random_start_level=False,
                         cameraWidth=200,
                         cameraHeight=60,
                         show_image_reconstruction=True)


# get first observation values
obs = env.reset()

# one class for all different types of hardcoded agents
if useTeleoperation:
    agent = HardcodedAgents(obs, True)
else:
    agent = HardcodedAgents(obs, False)

# Simulate the agent
for step in range(max_episode_length):
    # let the agent know the current timestep
    agent.step = step

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

    # Simulate the environment for one step
    agent.obs, agent.reward, agent.done, agent.info = env.step(actions)


    if agent.done:
        agent.obs = env.reset()

