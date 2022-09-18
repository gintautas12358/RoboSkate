# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nrp_sb3.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nrp_sb3.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rnrp_sb3.proto\"\n\n\x08No_Param\"g\n\x04\x44\x61ta\x12\x11\n\titeration\x18\x01 \x01(\x05\x12\x18\n\x10\x62oard_position_x\x18\x02 \x01(\x02\x12\x18\n\x10\x62oard_position_y\x18\x03 \x01(\x02\x12\x18\n\x10\x62oard_position_z\x18\x04 \x01(\x02\"<\n\x07\x41\x63tions\x12\x0f\n\x07\x61\x63tion1\x18\x01 \x01(\x02\x12\x0f\n\x07\x61\x63tion2\x18\x02 \x01(\x02\x12\x0f\n\x07\x61\x63tion3\x18\x03 \x01(\x02\x32\xa5\x01\n\x07Manager\x12\"\n\x0cget_NRP_data\x12\t.No_Param\x1a\x05.Data\"\x00\x12(\n\x0fget_SB3_actions\x12\t.No_Param\x1a\x08.Actions\"\x00\x12\"\n\x0cset_NRP_data\x12\x05.Data\x1a\t.No_Param\"\x00\x12(\n\x0fset_SB3_actions\x12\x08.Actions\x1a\t.No_Param\"\x00\x62\x06proto3'
)




_NO_PARAM = _descriptor.Descriptor(
  name='No_Param',
  full_name='No_Param',
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
  serialized_start=17,
  serialized_end=27,
)


_DATA = _descriptor.Descriptor(
  name='Data',
  full_name='Data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='iteration', full_name='Data.iteration', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='board_position_x', full_name='Data.board_position_x', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='board_position_y', full_name='Data.board_position_y', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='board_position_z', full_name='Data.board_position_z', index=3,
      number=4, type=2, cpp_type=6, label=1,
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
  serialized_start=29,
  serialized_end=132,
)


_ACTIONS = _descriptor.Descriptor(
  name='Actions',
  full_name='Actions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='action1', full_name='Actions.action1', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='action2', full_name='Actions.action2', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='action3', full_name='Actions.action3', index=2,
      number=3, type=2, cpp_type=6, label=1,
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
  serialized_start=134,
  serialized_end=194,
)

DESCRIPTOR.message_types_by_name['No_Param'] = _NO_PARAM
DESCRIPTOR.message_types_by_name['Data'] = _DATA
DESCRIPTOR.message_types_by_name['Actions'] = _ACTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

No_Param = _reflection.GeneratedProtocolMessageType('No_Param', (_message.Message,), {
  'DESCRIPTOR' : _NO_PARAM,
  '__module__' : 'nrp_sb3_pb2'
  # @@protoc_insertion_point(class_scope:No_Param)
  })
_sym_db.RegisterMessage(No_Param)

Data = _reflection.GeneratedProtocolMessageType('Data', (_message.Message,), {
  'DESCRIPTOR' : _DATA,
  '__module__' : 'nrp_sb3_pb2'
  # @@protoc_insertion_point(class_scope:Data)
  })
_sym_db.RegisterMessage(Data)

Actions = _reflection.GeneratedProtocolMessageType('Actions', (_message.Message,), {
  'DESCRIPTOR' : _ACTIONS,
  '__module__' : 'nrp_sb3_pb2'
  # @@protoc_insertion_point(class_scope:Actions)
  })
_sym_db.RegisterMessage(Actions)



_MANAGER = _descriptor.ServiceDescriptor(
  name='Manager',
  full_name='Manager',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=197,
  serialized_end=362,
  methods=[
  _descriptor.MethodDescriptor(
    name='get_NRP_data',
    full_name='Manager.get_NRP_data',
    index=0,
    containing_service=None,
    input_type=_NO_PARAM,
    output_type=_DATA,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='get_SB3_actions',
    full_name='Manager.get_SB3_actions',
    index=1,
    containing_service=None,
    input_type=_NO_PARAM,
    output_type=_ACTIONS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='set_NRP_data',
    full_name='Manager.set_NRP_data',
    index=2,
    containing_service=None,
    input_type=_DATA,
    output_type=_NO_PARAM,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='set_SB3_actions',
    full_name='Manager.set_SB3_actions',
    index=3,
    containing_service=None,
    input_type=_ACTIONS,
    output_type=_NO_PARAM,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_MANAGER)

DESCRIPTOR.services_by_name['Manager'] = _MANAGER

# @@protoc_insertion_point(module_scope)