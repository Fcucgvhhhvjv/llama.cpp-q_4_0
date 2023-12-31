#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

./main -m ./models/ggml-alpaca-7b-q4.bin \
        --color \
       -f ./prompts/alpaca.txt \
       --ctx_size 2048 \
       -n -1 \
       -ins -b 256 \
       --top_k 10000 \
       --temp 0.2 \
       --repeat_penalty 1.1 \
       -t 7
