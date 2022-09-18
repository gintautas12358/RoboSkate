"""
Multiprosessing learning environment for DummyBall with PPO
"""

from stable_baselines3 import A2C, PPO
from scripts.python_nrp.sb3.NRPDummyBallGymEnv_image import NRPDummyBallEnv

n_parallel_env = 1
agent_name = "PPO_agent_canny_image_cnn_policy_150"
render_image_width = 150
render_image_height = 150
base_Port = 50051
log_interval = 1

if __name__ == '__main__':

    env = NRPDummyBallEnv(render_image_width, render_image_height)

    log_dir = "./scripts/python_nrp/sb3/agent_models/" + agent_name

    model = PPO.load(log_dir + str(0), env)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info_list = env.step(action)

    # Clean up the environment’s resources.
    env.close()

