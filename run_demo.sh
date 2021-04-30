#!/bin/bash

# Check installation and activate source 
source env/bin/activate

#set omp and run demo with faulthandler
export OMP_NUM_THREADS=48
python3 -q -X faulthandler demo.py

#close source for safe exit
deactivate
