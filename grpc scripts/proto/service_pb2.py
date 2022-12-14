# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='service.proto',
  package='communication',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rservice.proto\x12\rcommunication\"!\n\x11InitializeRequest\x12\x0c\n\x04json\x18\x01 \x01(\t\"K\n\x0fInitializeReply\x12\x0f\n\x07success\x18\x01 \x01(\x0c\x12\x12\n\nimageWidth\x18\x02 \x01(\r\x12\x13\n\x0bimageHeight\x18\x03 \x01(\r\" \n\rShutdownReply\x12\x0f\n\x07success\x18\x01 \x01(\x0c\"l\n\x0cGetInfoReply\x12\x0f\n\x07success\x18\x01 \x01(\x0c\x12\x15\n\rboardPosition\x18\x02 \x03(\x02\x12\x15\n\rboardRotation\x18\x03 \x03(\x02\x12\x1d\n\x15\x62oardCraneJointAngles\x18\x04 \x03(\x02\"/\n\x0eSetInfoRequest\x12\x1d\n\x15\x62oardCraneJointAngles\x18\x01 \x03(\x02\"\x1f\n\x0cSetInfoReply\x12\x0f\n\x07success\x18\x01 \x01(\x0c\"4\n\x0eGetCameraReply\x12\x0f\n\x07success\x18\x01 \x01(\x0c\x12\x11\n\timageData\x18\x02 \x01(\x0c\"\x1e\n\x0eRunGameRequest\x12\x0c\n\x04time\x18\x01 \x01(\x02\";\n\x0cRunGameReply\x12\x0f\n\x07success\x18\x01 \x01(\x0c\x12\x1a\n\x12totalSimulatedTime\x18\x02 \x01(\x02\"\n\n\x08NoParams2\xcd\x03\n\x14\x43ommunicationService\x12P\n\ninitialize\x12 .communication.InitializeRequest\x1a\x1e.communication.InitializeReply\"\x00\x12\x43\n\x08shutdown\x12\x17.communication.NoParams\x1a\x1c.communication.ShutdownReply\"\x00\x12\x42\n\x08get_info\x12\x17.communication.NoParams\x1a\x1b.communication.GetInfoReply\"\x00\x12H\n\x08set_info\x12\x1d.communication.SetInfoRequest\x1a\x1b.communication.SetInfoReply\"\x00\x12\x46\n\nget_camera\x12\x17.communication.NoParams\x1a\x1d.communication.GetCameraReply\"\x00\x12H\n\x08run_game\x12\x1d.communication.RunGameRequest\x1a\x1b.communication.RunGameReply\"\x00\x62\x06proto3'
)




_INITIALIZEREQUEST = _descriptor.Descriptor(
  name='InitializeRequest',
  full_name='communication.InitializeRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='json', full_name='communication.InitializeRequest.json', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=65,
)


_INITIALIZEREPLY = _descriptor.Descriptor(
  name='InitializeReply',
  full_name='communication.InitializeReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='communication.InitializeReply.success', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='imageWidth', full_name='communication.InitializeReply.imageWidth', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='imageHeight', full_name='communication.InitializeReply.imageHeight', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=67,
  serialized_end=142,
)


_SHUTDOWNREPLY = _descriptor.Descriptor(
  name='ShutdownReply',
  full_name='communication.ShutdownReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='communication.ShutdownReply.success', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=144,
  serialized_end=176,
)


_GETINFOREPLY = _descriptor.Descriptor(
  name='GetInfoReply',
  full_name='communication.GetInfoReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='communication.GetInfoReply.success', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='boardPosition', full_name='communication.GetInfoReply.boardPosition', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='boardRotation', full_name='communication.GetInfoReply.boardRotation', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='boardCraneJointAngles', full_name='communication.GetInfoReply.boardCraneJointAngles', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=178,
  serialized_end=286,
)


_SETINFOREQUEST = _descriptor.Descriptor(
  name='SetInfoRequest',
  full_name='communication.SetInfoRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='boardCraneJointAngles', full_name='communication.SetInfoRequest.boardCraneJointAngles', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=288,
  serialized_end=335,
)


_SETINFOREPLY = _descriptor.Descriptor(
  name='SetInfoReply',
  full_name='communication.SetInfoReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='communication.SetInfoReply.success', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=337,
  serialized_end=368,
)


_GETCAMERAREPLY = _descriptor.Descriptor(
  name='GetCameraReply',
  full_name='communication.GetCameraReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='communication.GetCameraReply.success', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='imageData', full_name='communication.GetCameraReply.imageData', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=370,
  serialized_end=422,
)


_RUNGAMEREQUEST = _descriptor.Descriptor(
  name='RunGameRequest',
  full_name='communication.RunGameRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='time', full_name='communication.RunGameRequest.time', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=424,
  serialized_end=454,
)


_RUNGAMEREPLY = _descriptor.Descriptor(
  name='RunGameReply',
  full_name='communication.RunGameReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='communication.RunGameReply.success', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='totalSimulatedTime', full_name='communication.RunGameReply.totalSimulatedTime', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=456,
  serialized_end=515,
)


_NOPARAMS = _descriptor.Descriptor(
  name='NoParams',
  full_name='communication.NoParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=517,
  serialized_end=527,
)

DESCRIPTOR.message_types_by_name['InitializeRequest'] = _INITIALIZEREQUEST
DESCRIPTOR.message_types_by_name['InitializeReply'] = _INITIALIZEREPLY
DESCRIPTOR.message_types_by_name['ShutdownReply'] = _SHUTDOWNREPLY
DESCRIPTOR.message_types_by_name['GetInfoReply'] = _GETINFOREPLY
DESCRIPTOR.message_types_by_name['SetInfoRequest'] = _SETINFOREQUEST
DESCRIPTOR.message_types_by_name['SetInfoReply'] = _SETINFOREPLY
DESCRIPTOR.message_types_by_name['GetCameraReply'] = _GETCAMERAREPLY
DESCRIPTOR.message_types_by_name['RunGameRequest'] = _RUNGAMEREQUEST
DESCRIPTOR.message_types_by_name['RunGameReply'] = _RUNGAMEREPLY
DESCRIPTOR.message_types_by_name['NoParams'] = _NOPARAMS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InitializeRequest = _reflection.GeneratedProtocolMessageType('InitializeRequest', (_message.Message,), {
  'DESCRIPTOR' : _INITIALIZEREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.InitializeRequest)
  })
_sym_db.RegisterMessage(InitializeRequest)

InitializeReply = _reflection.GeneratedProtocolMessageType('InitializeReply', (_message.Message,), {
  'DESCRIPTOR' : _INITIALIZEREPLY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.InitializeReply)
  })
_sym_db.RegisterMessage(InitializeReply)

ShutdownReply = _reflection.GeneratedProtocolMessageType('ShutdownReply', (_message.Message,), {
  'DESCRIPTOR' : _SHUTDOWNREPLY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.ShutdownReply)
  })
_sym_db.RegisterMessage(ShutdownReply)

GetInfoReply = _reflection.GeneratedProtocolMessageType('GetInfoReply', (_message.Message,), {
  'DESCRIPTOR' : _GETINFOREPLY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.GetInfoReply)
  })
_sym_db.RegisterMessage(GetInfoReply)

SetInfoRequest = _reflection.GeneratedProtocolMessageType('SetInfoRequest', (_message.Message,), {
  'DESCRIPTOR' : _SETINFOREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.SetInfoRequest)
  })
_sym_db.RegisterMessage(SetInfoRequest)

SetInfoReply = _reflection.GeneratedProtocolMessageType('SetInfoReply', (_message.Message,), {
  'DESCRIPTOR' : _SETINFOREPLY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.SetInfoReply)
  })
_sym_db.RegisterMessage(SetInfoReply)

GetCameraReply = _reflection.GeneratedProtocolMessageType('GetCameraReply', (_message.Message,), {
  'DESCRIPTOR' : _GETCAMERAREPLY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.GetCameraReply)
  })
_sym_db.RegisterMessage(GetCameraReply)

RunGameRequest = _reflection.GeneratedProtocolMessageType('RunGameRequest', (_message.Message,), {
  'DESCRIPTOR' : _RUNGAMEREQUEST,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.RunGameRequest)
  })
_sym_db.RegisterMessage(RunGameRequest)

RunGameReply = _reflection.GeneratedProtocolMessageType('RunGameReply', (_message.Message,), {
  'DESCRIPTOR' : _RUNGAMEREPLY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.RunGameReply)
  })
_sym_db.RegisterMessage(RunGameReply)

NoParams = _reflection.GeneratedProtocolMessageType('NoParams', (_message.Message,), {
  'DESCRIPTOR' : _NOPARAMS,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:communication.NoParams)
  })
_sym_db.RegisterMessage(NoParams)



_COMMUNICATIONSERVICE = _descriptor.ServiceDescriptor(
  name='CommunicationService',
  full_name='communication.CommunicationService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=530,
  serialized_end=991,
  methods=[
  _descriptor.MethodDescriptor(
    name='initialize',
    full_name='communication.CommunicationService.initialize',
    index=0,
    containing_service=None,
    input_type=_INITIALIZEREQUEST,
    output_type=_INITIALIZEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='shutdown',
    full_name='communication.CommunicationService.shutdown',
    index=1,
    containing_service=None,
    input_type=_NOPARAMS,
    output_type=_SHUTDOWNREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='get_info',
    full_name='communication.CommunicationService.get_info',
    index=2,
    containing_service=None,
    input_type=_NOPARAMS,
    output_type=_GETINFOREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='set_info',
    full_name='communication.CommunicationService.set_info',
    index=3,
    containing_service=None,
    input_type=_SETINFOREQUEST,
    output_type=_SETINFOREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='get_camera',
    full_name='communication.CommunicationService.get_camera',
    index=4,
    containing_service=None,
    input_type=_NOPARAMS,
    output_type=_GETCAMERAREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='run_game',
    full_name='communication.CommunicationService.run_game',
    index=5,
    containing_service=None,
    input_type=_RUNGAMEREQUEST,
    output_type=_RUNGAMEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_COMMUNICATIONSERVICE)

DESCRIPTOR.services_by_name['CommunicationService'] = _COMMUNICATIONSERVICE

# @@protoc_insertion_point(module_scope)
