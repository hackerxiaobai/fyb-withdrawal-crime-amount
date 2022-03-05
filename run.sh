#!/bin/bash
export PYTHONPATH="$PWD"
python utils.py ./data/train.txt ./data/
python main.py