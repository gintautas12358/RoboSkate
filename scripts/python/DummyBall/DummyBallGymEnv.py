"""
This Gym Environment is the interface for DummyBall.
The Reward function is for now only the distance to the origin.

Creating Custom Gym Environments Tutorials
https://blog.paperspace.com/creating-custom-environments-openai-gym/
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb#scrollTo=rYzDXA9vJfz1

"""

import numpy as np
import gym
from gym import spaces
import grpc
import time
from scripts.python.grpcClient import service_pb2_grpc
from scripts.python.grpcClient.service_pb2 import InitializeRequest, NoParams, RunGameRequest, SetInfoRequest


# TODO: Should we pass the parameter when setting the environment?
use_camera = False

# The observation will be the board state information like position, velocity and angle
# Value Range for observations abs(-Min) = Max
max_Joint_force_1 = 100.0

# --------------------------------------------------------------------------------
# ------------------ gRPC functions ----------------------------------------------
# --------------------------------------------------------------------------------
def initialize(stub, string):
    reply = stub.initialize(InitializeRequest(json=string))
    if reply.success == bytes('0', encoding='utf8'):
        print("Initialize gRPC success")
        print(reply.imageWidth)
        print(reply.imageHeight)
        pass
    else:
        print("Initialize failure")


def set_info(stub, joint1, joint2, joint3):
    # passing random value to observe more interesting motions of the ball in the game
    reply = stub.set_info(SetInfoRequest( boardCraneJointAngles = [joint1 * max_Joint_force_1, 0, 0]))
    if reply.success == bytes('0', encoding='utf8'):
        #print("SetInfo success" + str(joint1))
        pass
    else:
        print("SetInfo gRPC failure")

def get_info(stub):
    reply = stub.get_info(NoParams())
    if reply.success == bytes('0', encoding='utf8'):
        # normalisation

        reply.boardPosition[0] *= 1
        reply.boardPosition[1] *= 1
        reply.boardPosition[2] *= 1

        return reply
    else:
        print("GetInfo gRPC failure")

def get_camera(stub, i):
    reply = stub.get_camera(NoParams())
    _retrieve_image(reply, i)

def _retrieve_image(reply, i):
    # how to access the image data from reply.imageData and store it in a file, as an example
    # create folder "images" in the python workind directory and uncomment the lines below if you want to save the file
    # TODO: Distinguish between images from different environments.
    with open('./images/received-image-{}.{}'.format(i, 'jpg'), 'wb') as image_file:
       image_file.write(reply.imageData)

def run_game(stub, simTime):
    reply = stub.run_game(RunGameRequest(time=simTime))
    if reply.success == bytes('0', encoding='utf8'):
        #print("Total simulated time:" + str(reply.totalSimulatedTime) )
        pass
    else:
        print("RunGame gRPC failure")

def shutdown(stub):
    reply = stub.shutdown(NoParams())
    if reply.success == bytes('0', encoding='utf8'):
        print("Shutdown gRPC success")
    else:
        print("Shutdown gRPC failure")

# --------------------------------------------------------------------------------
# ------------------ RoboSkate Environment ---------------------------------------
# --------------------------------------------------------------------------------


class DummyBallEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a env for the RoboSkate Unity game.
    """

    def __init__(self, image_width, image_height, Port):
        super(DummyBallEnv, self).__init__()
        print('init environment')

        self.imageHeight = image_height
        self.imageWidth = image_width

        # gRPC channel
        address = 'localhost:' + Port
        print(address)
        channel = grpc.insecure_channel(address)
        self.stub = service_pb2_grpc.CommunicationServiceStub(channel)

        # state from the game: position, velocity, angle
        self.state = 0
        self.reward = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # discrete actions: joint1, joint2, joint3
        # The first array are the lowest accepted values, the second are the highest accepted values.
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float)

        self.observation_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float)

    def reset(self):
        print('reset env')

        initialize(self.stub, "0," + str(self.imageWidth) + "," + str(self.imageHeight))
        self.start = time.time()
        self.stepcount = 0

        # get the current state
        self.state = get_info(self.stub)
        self.reward = self.get_reward(self.state)

        return np.array([self.state.boardPosition [0], self.state.boardPosition [1], self.state.boardPosition [2]]).astype(np.float)

    # Reward Function
    def get_reward(self, state):
        # biggest reward for doing nothing
        reward = self.state.boardPosition [2]
        return reward

    def step(self, action):
        # set the actions
        # The observation will be the board state information like position, velocity and angle

        set_info(self.stub, action[0], 0, 0)

        # Run physics simulation in Unity
        # TODO: Should we pass the parameter when setting the environment?
        run_game(self.stub, 0.2) # 0.2 Seconds Physical simulation step

        # get the current state
        self.state = get_info(self.stub)

        # get current Reward
        self.reward = self.get_reward(self.state)

        if use_camera:
            # render image in Unity
            get_camera(self.stub, self.stepcount)


        # TODO: How do we define a goal?
        # TODO: Do we need episodes to end for vectorized environments?
        if self.state.boardPosition[1] < -10:
            done = True
        else:
            done = False

        # Optionally we can pass additional info, we are not using that for now
        info = {"time": (time.time() - self.start)}

        self.stepcount += 1

        return np.array([self.state.boardPosition [0], self.state.boardPosition [1], self.state.boardPosition [2],]).astype(np.float), self.reward, done, info

    def render(self, mode='human'):
        print('render')
        # TODO: Can we use the function in a meaningful way?

    def close(self):
        print('close')
        shutdown(self.stub)
        self.end = time.time()
        print("Time for the epoch: " + str(self.end - self.start))
        pass
