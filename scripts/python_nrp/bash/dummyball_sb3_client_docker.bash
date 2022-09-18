#!/bin/bash

sudo docker exec -it nrp_game \
  bash -c "
  source /home/nrpuser/.bashrc;
  cd /home/nrpuser/nrp-core/examples/roboskate;
  python3 scripts/python_nrp/sb3/NRPDummyBallSB3_PPO.py
  "
