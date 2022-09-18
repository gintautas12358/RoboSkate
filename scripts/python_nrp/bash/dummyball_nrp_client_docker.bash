#!/bin/bash

sudo docker exec -it nrp_game \
  bash -c "
  source /home/nrpuser/.bashrc; cd /home/nrpuser/nrp-core/examples/roboskate;
  NRPSimulation -c scripts/python_nrp/simulation_config.json
  "

