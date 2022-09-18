from NRPPythonEngineModule import EngineScript,RegisterEngine

@RegisterEngine()
class Script(EngineScript):
    def initialize(self):
        self._registerDevice("JointAngles")
        self._registerDevice("device1")

        # gRPC
        import grpc
        port = 50055
        address = 'localhost:' + str(port)
        print(address)
        channel = grpc.insecure_channel(address)
        from scripts.python_nrp.grpc import nrp_sb3_pb2_grpc
        self.stub = nrp_sb3_pb2_grpc.ManagerStub(channel)

        # Joints need to be set
        self.setD(0, 0, 0)


    def runLoop(self, timestep):

        data = self.getD()
        state = data["board_position"]
        i = data["iteration"]

        from scripts.python_nrp.grpc.grpc_functions import set_NRP_data
        set_NRP_data(self.stub, i, state[2], state[0], state[1])
        print(f'ball i: {i} y: {state[0]} z: {state[1]} x: {state[2]}')

        from scripts.python_nrp.grpc.grpc_functions import get_SB3_actions
        actions = get_SB3_actions(self.stub)

        print(f'i: {i} force: {actions[0]}')
        self.setD(actions[0], 0, 0)


    def shutdown():
        pass

    def setD(self, j1, j2, j3):
        self._setDevice("JointAngles", { "joint_angles" : [j1, j2, j3]})

    def getD(self):
        data = self._getDevice("device1")
        return data

# EOF
