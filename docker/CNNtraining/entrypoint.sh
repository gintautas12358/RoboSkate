#!/bin/bash
# This script is the entrypoint for the Docker image.

cd /home/ubuntu/roboskate


# Run training
python3 -m scripts.python.RoboSkate.Image_Segmentation
