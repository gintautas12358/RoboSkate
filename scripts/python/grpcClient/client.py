from __future__ import print_function
import random
import logging
import time
import grpc
import service_pb2_grpc
from service_pb2 import InitializeRequest, NoParams, RunGameRequest, SetInfoRequest


def initialize(stub, string):
    reply = stub.initialize(InitializeRequest(json=string))
    if reply.success == bytes('0', encoding='utf8'):
        print("Initialize success")
        print(reply.imageWidth)
        print(reply.imageHeight)
    else:
        print("Initialize failure")


def shutdown(stub):
    reply = stub.shutdown(NoParams())
    if reply.success == bytes('0', encoding='utf8'):
        print("Shutdown success")
    else:
        print("Shutdown failure")


def get_camera(stub, i):
    reply = stub.get_camera(NoParams())
    _retrieve_image(reply, i)


def run_game(stub):
    reply = stub.run_game(RunGameRequest(time=0.2))
    if reply.success == bytes('0', encoding='utf8'):
        print("Total simulated time:" + str(reply.totalSimulatedTime) )
    else:
        print("RunGame failure")


def set_info(stub):
    # passing random value to observe more interesting motions of the ball in the game
    reply = stub.set_info(SetInfoRequest( boardCraneJointAngles = [(random.random() - 0.5) * 100, 10, 10]))
    if reply.success == bytes('0', encoding='utf8'):
        print("SetInfo success")
    else:
        print("SetInfo failure")


def get_info(stub):
    reply = stub.get_info(NoParams())
    # boardPosition(9 floats): position(m): x, y, z, linearVelocity.x, y, z (m/sec); angularVelocity x, y, z(rad/sec);
    # boardRotation(7 floats): Euler(deg): x, y, z; Quaternion x, y, z, w
    # boardCraneJointAngles(6 floats): Angles(deg): base, middle, top; Velocities(deg/sec) base, middle, top;
    if reply.success == bytes('0', encoding='utf8'):
        print("GetInfo position: " + str(reply.boardPosition))
        print("GetInfo rotation: " + str(reply.boardRotation))
        print("GetInfo craneAngles: " + str(reply.boardCraneJointAngles))
    else:
        print("GetInfo failure")


def _retrieve_image(reply, i):
    # how to access the image data from reply.imageData and store it in a file, as an example
    # create folder "images" in the python workind directory and uncomment the lines below if you want to save the file
    with open('images/received-image-{}.{}'.format(i, 'jpg'), 'wb') as image_file:
       image_file.write(reply.imageData)
    #pass



def run():

    # random ports are also supported, to make it possible to implement parallelization and connect to multiple games
    # on the same PC. For this the game should be started with "-p port_number" through cmd/shell
    port_number = 50051
    with grpc.insecure_channel('localhost:'+str(port_number)) as channel:
        stub = service_pb2_grpc.CommunicationServiceStub(channel)

        for _ in range(1):
            print("-------------- Initialize --------------")
            initialize(stub, "0,200,200")
            start = time.time()
            for i in range(0, 10):
                print(str(i) + "-------------- GetInfo --------------")
                get_info(stub)
                print(str(i) + "-------------- GetCamera --------------")
                get_camera(stub, i)
                print(str(i) + "-------------- SetInfo --------------")
                set_info(stub)
                print(str(i) + "-------------- RunGame --------------")
                run_game(stub)


            #print("-------------- Shutdown --------------")
            #shutdown(stub);
        end = time.time()
        print("Time it took: " + str(end - start))

if __name__ == '__main__':
    logging.basicConfig()
    run()
