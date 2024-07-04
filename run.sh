#!/usr/bin/env bash
make clean build

make run NUM_DATA="256" INPUT="input_256.txt" OUTPUT="output_256.txt"              
make run NUM_DATA="512" INPUT="input_512.txt" OUTPUT="output_512.txt"
make run NUM_DATA="1024" INPUT="input_1024.txt" OUTPUT="output_1024.txt"
make run NUM_DATA="4096" INPUT="input_4096.txt" OUTPUT="output_4096.txt"
make run NUM_DATA="10000" INPUT="input_10000.txt" OUTPUT="output_10000.txt"
make run NUM_DATA="30000" INPUT="input_30000.txt" OUTPUT="output_30000.txt"