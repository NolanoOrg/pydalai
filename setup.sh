#!/bin/sh
conda create -n pydalai python==3.10.9
conda activate pydalai
pip install -r requirments.txt
python3 setup.py
cd llama.cpp
make
python3 convert-pth-to-ggml.py models/7B/ 1
./quantize.sh 7B
cd ..
python3 server.py