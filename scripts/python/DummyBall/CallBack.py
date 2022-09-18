import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

# --------------------------------------------------------------------------------
# ------------------ Tensorboard Callback ----------------------------------------
# --------------------------------------------------------------------------------
# CMD/Terminal command: $ tensorboard --logdir ./scripts/python/tensorboard/agent_name/
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, n_parallel_env, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.n_parallel_env = n_parallel_env

    def _on_step(self) -> bool:
        # Log values
        # TODO: implement for multiple environments
        if self.n_parallel_env == 1:
            self.logger.record('Joint/Angle/1', self.training_env.envs[0].state.boardCraneJointAngles[0])
            self.logger.record('Joint/location/1', self.training_env.envs[0].state.boardPosition[2])
            self.logger.record('Reward/Total' , self.training_env.envs[0].total_reward)
        return True


# --------------------------------------------------------------------------------
# ------------------ Save On Best Training Reward Callback ----------------------------------------
# --------------------------------------------------------------------------------
# 4_callbacks_hyperparameter_tuning.ipynb

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True
