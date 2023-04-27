#!/bin/bash

# cd to scripts dir
cd `dirname $0`

# download model to models dir
echo "Downloading model"
python ./convert_gptneox_to_ggml.py OpenAssistant/stablelm-7b-sft-v7-epoch-3 ../models/stablelm

# remove temp cache dir
echo "Removing temp cache dir"
rm -r ../models/stablelm-cache

# quantize model
echo "Quantizing model"
cd ..
python ./scripts/quantize-gptneox.py ./models/stablelm/ggml-stablelm-7b-sft-v7-epoch-3-f16.bin

# remove non-quantized model
echo "Remove non-quantized model"
rm ./models/stablelm/ggml-stablelm-7b-sft-v7-epoch-3-f16.bin

# done!
cd scripts
echo "Done. Run 'chat-stablelm-7B.sh' to test model."
