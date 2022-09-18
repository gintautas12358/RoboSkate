import cv2
import numpy as np

# https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
# https://www.sicara.ai/blog/2019-03-12-edge-detection-in-opencv

def preprocess_image(step) -> np.ndarray:

    frame = cv2.imread('./images/received-image-' + str(step) + '.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #frame = cv2.resize(frame, (20, 20), interpolation=cv2.INTER_AREA)

    # method 1: Edge detection
    frame = cv2.Canny(frame, 35, 120)
    # method 2: Threshold
    #frame = cv2.threshold(frame, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    frame = cv2.resize(frame, (20, 20), interpolation=cv2.INTER_AREA)

    cv2.imshow(str(i), frame)

    return frame[..., None]

for i in range(100,300,27):
    preprocess_image(i)

cv2.waitKey(0)
cv2.destroyAllWindows()
