#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

./main-oasst -m ./models/pythia/ggml-oasst-sft-4-pythia-12b-epoch-3.5-q4_0.bin \
        -c 2048 \
        -b 8 \
        -n -1 \
        -t 8 \
        --instruct \
        --color \
        --top_k 30 \
        --top_p 0.95 \
        --temp 0.8 \
        --repeat_last_n 64 \
        --repeat_penalty 1.1 \
        --seed 0
