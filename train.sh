#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

#rm -rf __pycache__
#python3 trainMain.py --datadir ./dataset --batchSize 8 --nepochs 50 --classes 25 --model resnet50

#rm -rf __pycache__
#python3 trainMain.py --datadir ./dataset --batchSize 8 --nepochs 50 --classes 25 --model densenet201

rm -rf __pycache__
python3 trainMain.py --datadir ./dataset --batchSize 8 --nepochs 50 --classes 25 --model inception
