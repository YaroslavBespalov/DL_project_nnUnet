#!/usr/bin/env bash

PYTHONPATH=./ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python ./src/inference.py\
    --config=./configs/config_inference.yml\
    --paths=./configs/path.yml
