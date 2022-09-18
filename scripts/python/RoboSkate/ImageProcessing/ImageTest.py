import numpy as np
import gym
import cv2
from gym import spaces
import math
import grpc
import time
import io

from matplotlib import pyplot, pyplot as plt
from scipy import ndimage as ndi
from PIL import Image
from scripts.python.grpcClient import service_pb2_grpc
from scripts.python.grpcClient.service_pb2 import InitializeRequest, NoParams, RunGameRequest, SetInfoRequest
from skimage.util import random_noise

# Method for preprocessing the image
# input: reply, the image received in the retrieve_image function from the environment, in order to not save all
# the screenshots
# alternative, i, which is just a counting variable to open the images one after another, for this the images
# have to be saved locally
# left_obstacle, right_obstacle, front_obstacle are arrays, where the distances to the obstacles are saved
# output: filled arrays: left_obstacle, right_obstacle, front_obstacle
def preprocessing3(reply, i, left_obstacle,right_obstacle, front_obstacle):
    image = reply.imageData
    stream = io.BytesIO(image)
    img = Image.open(stream)

    #alternative with opening the saved screenshots
    #image = Image.open(r"C:/Users/mibet/PycharmProjects/G1_RoboSkate/images/RoboSkate-" + str(i) + ".jpg")

    #methods to get the separation of obstacles and free space
    dst = cv2.GaussianBlur(np.float32(img), (3, 3), 0)
    gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray.astype('uint8'), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    image = cv2.dilate(opening, kernel, iterations=2)
    image = ndi.rotate(image, 15, mode = 'constant')
    image = ndi.gaussian_filter(image, 2)
    image = random_noise(image, mode='speckle', mean=0.5)

    #saving the images for illustration purposes
    pyplot.imsave('C:/Users/mibet/PycharmProjects/G1_RoboSkate/images/received-image-' + str(i) + '.jpg', image)
    img = Image.open(r"C:/Users/mibet/PycharmProjects/G1_RoboSkate/images/received-image-" + str(i) + ".jpg")

    #coordinates of the robot
    robot_x = 50
    robot_y = 75
    img = img.load()

    #determinining the distance to the left obstacle for each pixel in a certain range
    for j in range (0,robot_x-20):
        if img[j, robot_y-20][1] < 20:
            left_obstacle.append(robot_x-j)
            break
        if j == robot_x-21:
            left_obstacle.append(99)

    # determinining the distance to the right obstacle for each pixel in a certain range
    for j in range(99, robot_x-20, -1):
        if img[j, robot_y-20][1] < 20:
            right_obstacle.append(j)
            break
        if j == robot_y-21:
            right_obstacle.append(99)

    # determinining the distance to the front obstacle for each pixel in a certain range
    for j in range(robot_y-40, 0,-1):
        if img[robot_x, j][1] > 20:
            front_obstacle.append(robot_y-j)
            break
        if j == 0:
            front_obstacle.append(75)

    return left_obstacle, right_obstacle, front_obstacle


#Creating graphs for the calculated distances
def graphs(left_obstacle, right_obstacle, front_obstacle):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
    ax[0].plot(left_obstacle)
    ax[0].set_title("Distance to left_obstacle")
    ax[1].plot(right_obstacle)
    ax[1].set_title("Distance to right_obstacle")
    ax[2].plot(front_obstacle)
    ax[2].set_title("Distance to front_obstacle")
    plt.show()


if __name__ == '__main__':
    left_obstacle = []
    right_obstacle = []
    front_obstacle = []
    for i in range(0, 557):
        left_obstacle,right_obstacle, front_obstacle = preprocessing3(i, left_obstacle, right_obstacle, front_obstacle)
    graphs(left_obstacle, right_obstacle, front_obstacle)
