"""
This Gym Environment is the interface for DummyBall.
The Reward function is for now only the distance to the origin.

Creating Custom Gym Environments Tutorials
https://blog.paperspace.com/creating-custom-environments-openai-gym/
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb#scrollTo=rYzDXA9vJfz1

"""
from io import StringIO

import PIL
import cv2
import numpy as np
import gym
from gym import spaces
import grpc
import time
import io
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.util import random_noise
from skimage import feature, img_as_float

from PIL import Image

from matplotlib import pyplot

from scripts.python.grpcClient import service_pb2_grpc
from scripts.python.grpcClient.service_pb2 import InitializeRequest, NoParams, RunGameRequest, SetInfoRequest

# TODO: Should we pass the parameter when setting the environment?
use_camera = True

# The observation will be the images taken from the DummyBall game.

# Left/right rotation strength
max_Joint_force_1 = 50.0


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
    # Only joint 1 is used for the DummyBall
    reply = stub.set_info(SetInfoRequest(boardCraneJointAngles=[joint1 * max_Joint_force_1, 0, 0]))
    if reply.success == bytes('0', encoding='utf8'):
        # print("SetInfo success" + str(joint1))
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
    # _retrieve_image(reply, i)
    return preprocessing3(reply, i)


def _retrieve_image(reply, i):
    # how to access the image data from reply.imageData and store it in a file, as an example
    # create folder "images" in the python workind directory and uncomment the lines below if you want to save the file
    # TODO: Distinguish between images from different environments.
    with open('./images/received-image-{}.{}'.format(i, 'jpg'), 'wb') as image_file:
        image_file.write(reply.imageData)


def run_game(stub, simTime):
    reply = stub.run_game(RunGameRequest(time=simTime))
    if reply.success == bytes('0', encoding='utf8'):
        # print("Total simulated time:" + str(reply.totalSimulatedTime) )
        pass
    else:
        print("RunGame gRPC failure")


def shutdown(stub):
    reply = stub.shutdown(NoParams())
    if reply.success == bytes('0', encoding='utf8'):
        print("Shutdown gRPC success")
    else:
        print("Shutdown gRPC failure")


def preprocessing3(reply, i):
    image = reply.imageData
    stream = io.BytesIO(image)
    img = Image.open(stream)
    dst = cv2.GaussianBlur(np.float32(img), (3, 3), 0)
    # dst = cv2.GaussianBlur(np.float32(image), (3, 3), 0)  # original 5,5
    gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray.astype('uint8'), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)  # original 3,3
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # ursprünglich 2
    image = cv2.dilate(opening, kernel, iterations=2)  # ursprünglich 3

    # image = ndi.rotate(image, 15, mode = 'constant')

    image = ndi.gaussian_filter(image, 4)  # original 4

    image = random_noise(image, mode='speckle', mean=1)
    # edges1 = feature.canny(image)
    #    edges2 = feature.canny(image, sigma=2)

    # if(step == 70 or step == 400):
    #   im = Image.open(r"C:/Users/mibet/PycharmProjects/G1_RoboSkate/images/received-image-" + str(step) + ".jpg")

    #  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    # ax[0].imshow(im)
    # ax[0].set_title('original', fontsize=20)
    # ax[1].imshow(image, cmap='gray')
    # ax[1].set_title(r'Canny filter, $\sigma=1, mean = 2$', fontsize=20)
    # plt.show()

    #pyplot.imsave('C:/Users/mibet/PycharmProjects/G1_RoboSkate/images/received-image-' + str(i) + '.jpg', image)
    #print("step")
    return image[..., None]


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
        self.start = time.time()
        self.stepcount = 0
        self.total_reward = 0
        # Frame Skipping
        self.frame_skip = 4

        # Save last n commands (throttle + steering)
        self.n_commands = 1
        n_command_history = 1
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_command_history = n_command_history

        # Custom frame-stack
        self.n_stack = 2
        self.stacked_obs = None

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

        initialize(self.stub, "0," + str(self.imageWidth) + "," + str(self.imageHeight))
        self.start = time.time()
        self.stepcount = 0
        self.total_reward = 0

        # get the current state
        self.state = get_info(self.stub)
        self.reward = self.get_reward()

        # render image in Unity
        image = get_camera(self.stub, self.stepcount)

        return image  # self.preprocess_image().astype(np.uint8)

    def step(self, action):
        # set the actions
        # The observation will be the images from the DummyBall game

        set_info(self.stub, action[0], 0, 0)

        # Run physics simulation in Unity
        # TODO: Should we pass the parameter when setting the environment?
        run_game(self.stub, 0.2)  # 0.2 Seconds Physical simulation step

        # get the current state
        self.state = get_info(self.stub)

        # render image in Unity
        image = get_camera(self.stub, self.stepcount)

        # Episode terminates, when falling or a certain mark is reached
        if self.get_corrected_z() < -1 or self.state.boardPosition[2] > 60:
            done = True
        else:
            done = False

        # Optionally we can pass additional info, we are not using that for now
        info = {"time": (time.time() - self.start)}

        self.stepcount += 1
        self.reward = self.get_reward()
        self.total_reward += self.reward

        return image, self.get_reward(), done, info

    def render(self, mode='human'):
        print('render')
        # TODO: Can we use the function in a meaningful way?

    def close(self):
        print('close')
        shutdown(self.stub)
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
        elif self.state.boardPosition[2] > 59:
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
        # frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        # Preprocessing methods:

        # method 1: Edge detection
        frame = cv2.Canny(frame, 35, 100)
        # method 2: Threshold
        # frame = cv2.threshold(frame, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        return frame[..., None]

    def preprocessing2(self) -> np.ndarray:
        # Step can not be negative
        if self.stepcount - 1 < 0:
            step = 0
        else:
            step = self.stepcount - 1
        # print("start")
        image = PIL.Image.open(
            'C:/Users/mibet/PycharmProjects/G1_RoboSkate/images/received-image-' + str(step) + '.jpg')
        dst = cv2.GaussianBlur(np.float32(image), (3, 3), 0)  # original 5,5
        gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray.astype('uint8'), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)  # original 3,3
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # ursprünglich 2
        image = cv2.dilate(opening, kernel, iterations=2)  # ursprünglich 3

        # image = ndi.rotate(image, 15, mode = 'constant')

        image = ndi.gaussian_filter(image, 4)  # original 4

        image = random_noise(image, mode='speckle', mean=1)
        # edges1 = feature.canny(image)
        edges2 = feature.canny(image, sigma=2)

        # if(step == 70 or step == 400):
        #   im = Image.open(r"C:/Users/mibet/PycharmProjects/G1_RoboSkate/images/received-image-" + str(step) + ".jpg")

        #  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
        # ax[0].imshow(im)
        # ax[0].set_title('original', fontsize=20)
        # ax[1].imshow(image, cmap='gray')
        # ax[1].set_title(r'Canny filter, $\sigma=1, mean = 2$', fontsize=20)
        # plt.show()

        pyplot.imsave('C:/Users/mibet/PycharmProjects/G1_RoboSkate/images/received-image-' + str(step) + '.jpg', image)
        print("step")
        return image[..., None]

    def get_corrected_z(self):
        # Platform is not horizontal. The platform follows f = kx + b

        k = -0.075584553
        b = -0.098834213 - 1e-6
        return self.state.boardPosition[1] - (k * self.state.boardPosition[2] + b)
