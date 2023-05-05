#!/bin/bash

# cd to scripts dir
cd `dirname $0`

# download model to models dir
echo "Downloading model"
python ./convert_gptneox_to_ggml.py togethercomputer/GPT-NeoXT-Chat-Base-20B ../models/chat-base

# remove temp cache dir
echo "Removing temp cache dir"
rm -r ../models/chat-base-cache

# quantize model
echo "Quantizing model"
cd ..
python ./scripts/quantize-gptneox.py ./models/chat-base/ggml-GPT-NeoXT-Chat-Base-20B-f16.bin

# remove non-quantized model
echo "Remove non-quantized model"
rm ./models/chat-base/ggml-GPT-NeoXT-Chat-Base-20B-f16.bin

# done!
cd scripts
echo "Done."
