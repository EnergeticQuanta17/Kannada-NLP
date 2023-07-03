#!/bin/bash

for i in {1..10}
do
    clear
    python get_input.py $i
    python predict.py
done
