#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=1 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=1 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=2 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=2 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=2 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=6 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=6 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=6 python3 my_tune2.py &
sleep 5
CUDA_VISIBLE_DEVICES=7 python3 my_tune2.py &
