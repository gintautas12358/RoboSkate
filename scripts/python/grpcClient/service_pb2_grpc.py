# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from scripts.python.grpcClient import service_pb2 as service__pb2


class CommunicationServiceStub(object):
    """The service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.initialize = channel.unary_unary(
                '/communication.CommunicationService/initialize',
                request_serializer=service__pb2.InitializeRequest.SerializeToString,
                response_deserializer=service__pb2.InitializeReply.FromString,
                )
        self.shutdown = channel.unary_unary(
                '/communication.CommunicationService/shutdown',
                request_serializer=service__pb2.NoParams.SerializeToString,
                response_deserializer=service__pb2.ShutdownReply.FromString,
                )
        self.get_info = channel.unary_unary(
                '/communication.CommunicationService/get_info',
                request_serializer=service__pb2.NoParams.SerializeToString,
                response_deserializer=service__pb2.GetInfoReply.FromString,
                )
        self.set_info = channel.unary_unary(
                '/communication.CommunicationService/set_info',
                request_serializer=service__pb2.SetInfoRequest.SerializeToString,
                response_deserializer=service__pb2.SetInfoReply.FromString,
                )
        self.get_camera = channel.unary_unary(
                '/communication.CommunicationService/get_camera',
                request_serializer=service__pb2.NoParams.SerializeToString,
                response_deserializer=service__pb2.GetCameraReply.FromString,
                )
        self.run_game = channel.unary_unary(
                '/communication.CommunicationService/run_game',
                request_serializer=service__pb2.RunGameRequest.SerializeToString,
                response_deserializer=service__pb2.RunGameReply.FromString,
                )


class CommunicationServiceServicer(object):
    """The service definition.
    """

    def initialize(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def shutdown(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_info(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def set_info(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def get_camera(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def run_game(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CommunicationServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'initialize': grpc.unary_unary_rpc_method_handler(
                    servicer.initialize,
                    request_deserializer=service__pb2.InitializeRequest.FromString,
                    response_serializer=service__pb2.InitializeReply.SerializeToString,
            ),
            'shutdown': grpc.unary_unary_rpc_method_handler(
                    servicer.shutdown,
                    request_deserializer=service__pb2.NoParams.FromString,
                    response_serializer=service__pb2.ShutdownReply.SerializeToString,
            ),
            'get_info': grpc.unary_unary_rpc_method_handler(
                    servicer.get_info,
                    request_deserializer=service__pb2.NoParams.FromString,
                    response_serializer=service__pb2.GetInfoReply.SerializeToString,
            ),
            'set_info': grpc.unary_unary_rpc_method_handler(
                    servicer.set_info,
                    request_deserializer=service__pb2.SetInfoRequest.FromString,
                    response_serializer=service__pb2.SetInfoReply.SerializeToString,
            ),
            'get_camera': grpc.unary_unary_rpc_method_handler(
                    servicer.get_camera,
                    request_deserializer=service__pb2.NoParams.FromString,
                    response_serializer=service__pb2.GetCameraReply.SerializeToString,
            ),
            'run_game': grpc.unary_unary_rpc_method_handler(
                    servicer.run_game,
                    request_deserializer=service__pb2.RunGameRequest.FromString,
                    response_serializer=service__pb2.RunGameReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'communication.CommunicationService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class CommunicationService(object):
    """The service definition.
    """

    @staticmethod
    def initialize(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.CommunicationService/initialize',
            service__pb2.InitializeRequest.SerializeToString,
            service__pb2.InitializeReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def shutdown(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.CommunicationService/shutdown',
            service__pb2.NoParams.SerializeToString,
            service__pb2.ShutdownReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_info(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.CommunicationService/get_info',
            service__pb2.NoParams.SerializeToString,
            service__pb2.GetInfoReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def set_info(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.CommunicationService/set_info',
            service__pb2.SetInfoRequest.SerializeToString,
            service__pb2.SetInfoReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def get_camera(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.CommunicationService/get_camera',
            service__pb2.NoParams.SerializeToString,
            service__pb2.GetCameraReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def run_game(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/communication.CommunicationService/run_game',
            service__pb2.RunGameRequest.SerializeToString,
            service__pb2.RunGameReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
