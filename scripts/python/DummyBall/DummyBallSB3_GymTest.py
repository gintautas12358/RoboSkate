"""
simple function to test the gym environment and move the agent hardcoded
"""
import numpy as np
from stable_baselines3.common.env_checker import check_env
from DummyBallGymEnv import DummyBallEnv
import math

env = DummyBallEnv()

# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

obs = env.reset()
print(env.observation_space)
print(env.action_space)

# Hardcoded agent
step = 0
done = False
while not done:
  step += 1
  print("Step {}".format(step))

  #Turn with 5
  obs, reward, done, info = env.step(np.array([5]))
  print(obs)
  print('reward=', reward)
