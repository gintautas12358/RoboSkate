"""
This Gym Environment is the interface for DummyBall.
The Reward function is for now only the distance to the origin.

Creating Custom Gym Environments Tutorials
https://blog.paperspace.com/creating-custom-environments-openai-gym/
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb#scrollTo=rYzDXA9vJfz1

"""
import cv2
import numpy as np
import gym
from gym import spaces
import grpc
import time

from scripts.python_nrp.grpc import nrp_sb3_pb2_grpc
from scripts.python_nrp.grpc.grpc_functions import get_NRP_data, set_SB3_actions

use_camera = True

# The observation will be the images taken from the DummyBall game.

# Left/right rotation strength
max_Joint_force_1 = 50.0


# --------------------------------------------------------------------------------
# ------------------ gRPC functions ----------------------------------------------
# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
# ------------------ DummyBall Environment ---------------------------------------
# --------------------------------------------------------------------------------

class NRPDummyBallEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a env for the RoboSkate Unity game.
    """

    def __init__(self, image_width, image_height):
        super(NRPDummyBallEnv, self).__init__()
        print('init environment')

        self.imageHeight = image_height
        self.imageWidth = image_width

        # gRPC channel
        port = 50055
        address = 'localhost:' + str(port)
        print(address)
        channel = grpc.insecure_channel(address)
        self.stub = nrp_sb3_pb2_grpc.ManagerStub(channel)

        # state from the game: position, velocity, angle
        self.reward = 0
        self.start = time.time()
        self.stepcount = 0
        self.total_reward = 0
        self.boardPosition = [0, 0, 0]

        # Define action and observation space
        # They must be gym.spaces objects
        # Continues actions: joint1
        # The first array are the lowest accepted values, the second are the highest accepted values.
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float)

        # Using images for observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(150, 150, 1), dtype=np.uint8)

    def reset(self):
        print('reset env')

        self.start = time.time()
        self.total_reward = 0

        # get the current state
        self.stepcount, board_position = get_NRP_data(self.stub)

        # waiting for NRP
        while self.stepcount == -1:
            self.stepcount, board_position = get_NRP_data(self.stub)
            time.sleep(0.1)

        self.save_unity_board_position(board_position)

        return self.preprocess_image().astype(np.uint8)

    def step(self, action):
        # set the actions
        # The observation will be the images from the DummyBall game

        set_SB3_actions(self.stub, action[0]*max_Joint_force_1, 0, 0)

        time.sleep(0.2)

        # get the current state
        self.stepcount, board_position = get_NRP_data(self.stub)
        print(f'step count: {self.stepcount}')

        self.save_unity_board_position(board_position)

        # Episode terminates, when falling or a certain mark is reached
        if self.get_corrected_z() < -1 or self.boardPosition[2] > 60:
            done = True
        else:
            done = False

        # Optionally we can pass additional info, we are not using that for now
        info = {"time": (time.time() - self.start)}

        self.reward = self.get_reward()
        self.total_reward += self.reward

        return self.preprocess_image(), self.get_reward(), done, info

    def render(self, mode='human'):
        print('render')
        # TODO: Can we use the function in a meaningful way?

    def close(self):
        print('close')

        self.end = time.time()
        print("Time for the epoch: " + str(self.end - self.start))
        pass

    # ======================================================================
    #               Additional utility functions
    # ======================================================================

    # Reward Function
    def get_reward(self):
        # Reward survival. Penalize falling.

        if self.get_corrected_z() < 0:
            reward = -10
        elif self.boardPosition[2] > 59:
            reward = 10
        else:
            reward = 1

        return reward

    # https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
    # https://www.sicara.ai/blog/2019-03-12-edge-detection-in-opencv

    def preprocess_image(self) -> np.ndarray:

        # Step can not be negative
        if self.stepcount - 1 < 0:
            step = 0
        else:
            step = self.stepcount - 1

        # Load image of the current step
        frame = cv2.imread('./images/received-image-' + str(step) + '.jpg')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # If resizing is need
        frame = cv2.resize(frame, (150, 150), interpolation=cv2.INTER_AREA)

        # Preprocessing methods:

        # method 1: Edge detection
        frame = cv2.Canny(frame, 35, 100)
        # method 2: Threshold
        #frame = cv2.threshold(frame, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        return frame[..., None]

    def get_corrected_z(self):
        # Platform is not horizontal. The platform follows f = kx + b

        k = -0.075584553
        b = -0.098834213 - 1e-6
        return self.boardPosition[1] - (k * self.boardPosition[2] + b)

    def save_unity_board_position(self, board_position):
        self.boardPosition[0] = board_position[1]
        self.boardPosition[1] = board_position[2]
        self.boardPosition[2] = board_position[0]

