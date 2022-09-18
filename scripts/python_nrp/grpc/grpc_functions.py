from time import sleep

import grpc

from scripts.python_nrp.grpc import nrp_sb3_pb2_grpc, nrp_sb3_pb2


def get_NRP_data(stub):
    request = nrp_sb3_pb2.No_Param()
    response = stub.get_NRP_data(request)
    return response.iteration, [response.board_position_x, response.board_position_y, response.board_position_z]

def set_NRP_data(stub, i, x, y, z):
    request = nrp_sb3_pb2.Data()
    request.iteration = i
    request.board_position_x = x
    request.board_position_y = y
    request.board_position_z = z
    stub.set_NRP_data(request)
    return nrp_sb3_pb2.No_Param()

def get_SB3_actions(stub):
    request = nrp_sb3_pb2.No_Param()
    response = stub.get_SB3_actions(request)
    return [response.action1, response.action2, response.action3]

def set_SB3_actions(stub, joint1, joint2, joint3):
    # Only joint 1 is used for the DummyBall
    request = nrp_sb3_pb2.Actions()
    request.action1 = joint1
    request.action2 = joint2
    request.action3 = joint3
    stub.set_SB3_actions(request)
    return nrp_sb3_pb2.No_Param()

# For testing the functionality
def main():
    # gRPC channel
    port = 50051
    address = 'localhost:' + str(port)
    print(address)
    channel = grpc.insecure_channel(address)
    stub = nrp_sb3_pb2_grpc.ManagerStub(channel)

    set_NRP_data(stub, 11, 47.5, 11.1, 45.9)
    i, position = get_NRP_data(stub)
    print(i, position)

    set_SB3_actions(stub, 10.0, 20.0, 30.0)
    actions = get_SB3_actions(stub)
    print(actions)

if __name__ == '__main__':
    main()
