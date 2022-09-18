import random

from NRPPythonModule import *
from NRPUnityGrpcEnginePython import *

def get_info(info):
    print("Board position: " + str(info.board_position))
    print("Board rotation: " + str(info.board_rotation))
    print("Joint angles: " + str(info.joint_angles))

def get_camera(camera, i):
    _retrieve_image(camera, i)

def _retrieve_image(reply, i):
    # how to access the image data from reply.imageData and store it in a file, as an example
    # create folder "images" in the python workind directory and uncomment the lines below if you want to save the file
    with open('images/received-image-{}.{}'.format(i, 'jpg'), 'wb') as image_file:
        image_file.write(reply.image_data)
    pass

iteration = 0

@FromEngineDevice(keyword='info', id=DeviceIdentifier('GetInfo', 'unity_grpc'))
@FromEngineDevice(keyword='camera', id=DeviceIdentifier('GetCamera', 'unity_grpc'))
@TransceiverFunction("python")
def transceiver_function(info, camera):

    # Process data coming from the game
    global iteration
    #get_info(info)
    get_camera(camera, iteration)

    # Send processed data to python script
    device1 = PythonDevice("device1", "python")

    data = {}
    data["board_position"] = info.board_position.tolist()
    data["iteration"] = iteration
    device1.data = data

    iteration += 1

    return [device1]
# EOF
