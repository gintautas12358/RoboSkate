import os
import cv2


def flip_some_images():
    directory = 'C:/Users/meric/Desktop/TUM/CBMLR/Repo/G1_RoboSkate/scripts/python/RoboSkateIL/RoboSkateExpertData/directionError/images/'

    os.chdir('C:/Users/meric/Desktop/TUM/CBMLR/Repo/G1_RoboSkate/scripts/python/RoboSkateIL/RoboSkateExpertData/directionError/images/')
    for file in os.listdir(directory):
        img = cv2.imread(directory + file)
        horizontal_img = cv2.flip(img, 1)

        # saving now
        cv2.imwrite('flipped' + file, horizontal_img)


if __name__ == '__main__':
    flip_some_images()
