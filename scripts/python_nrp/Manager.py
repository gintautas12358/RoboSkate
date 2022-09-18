from concurrent import futures

import grpc

from scripts.python_nrp.grpc import nrp_sb3_pb2_grpc, nrp_sb3_pb2


class Manager(nrp_sb3_pb2_grpc.ManagerServicer):

    def __init__(self):
        print("server is on")
        self.nrp_iteration = -1
        self.nrp_board_position = [-1, -1, -1]
        self.sb3_actions = [-1, -1, -1]


    def get_NRP_data(self, request, context):
        response = nrp_sb3_pb2.Data()
        response.iteration = self.nrp_iteration
        response.board_position_x = self.nrp_board_position[0]
        response.board_position_y = self.nrp_board_position[1]
        response.board_position_z = self.nrp_board_position[2]
        return response

    def get_SB3_actions(self, request, context):
        response = nrp_sb3_pb2.Actions()
        response.action1 = self.sb3_actions[0]
        response.action2 = self.sb3_actions[1]
        response.action3 = self.sb3_actions[2]
        return response

    def set_NRP_data(self, request, context):
        self.nrp_iteration = request.iteration
        self.nrp_board_position[0] = request.board_position_x
        self.nrp_board_position[1] = request.board_position_y
        self.nrp_board_position[2] = request.board_position_z
        print(f'set iteration: {self.nrp_iteration} position: {self.nrp_board_position}')
        return nrp_sb3_pb2.No_Param()

    def set_SB3_actions(self, request, context):
        print(f'set action1: {request.action1} action2: {request.action2} action3: {request.action3}')
        self.sb3_actions = [request.action1, request.action2, request.action3]
        return nrp_sb3_pb2.No_Param()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nrp_sb3_pb2_grpc.add_ManagerServicer_to_server(Manager(), server)
    server.add_insecure_port('[::]:50055')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

