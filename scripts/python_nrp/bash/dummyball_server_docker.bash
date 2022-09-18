#!/bin/bash

sudo docker rm nrp_game
xhost + local:

sudo docker run --runtime=nvidia -e DISPLAY \
  -v $PWD:/home/nrpuser/nrp-core/examples/roboskate:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --privileged -it \
  --name nrp_game nrp \
  bash -c "
  source /home/nrpuser/.bashrc;
  cd /home/nrpuser/nrp-core/examples/roboskate/;
  python3 scripts/python_nrp/Manager.py
  "

