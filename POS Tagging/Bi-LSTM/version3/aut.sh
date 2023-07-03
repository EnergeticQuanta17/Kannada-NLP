#!/bin/bash

for i in $(seq 1 10)
do
    clear
    python get_input.py $i
    python predict.py
done
