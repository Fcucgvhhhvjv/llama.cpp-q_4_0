#!/bin/bash

# Install eninops if not installed
pip install einops

# cd to scripts dir
cd `dirname $0`

# download model to models dir
echo "Downloading model"
python ./convert-bonsai-to-ggml.py byroneverson/bonsai-1b ../../models/bonsai

# remove temp cache dir
echo "Removing temp cache dir"
#rm -r ../../models/bonsai-cache

# quantize model
echo "Quantizing model"
cd ../..
python ./scripts/bonsai/quantize-bonsai.py ./models/bonsai/ggml-bonsai-1b-f16.bin

# remove non-quantized model
echo "Remove non-quantized model"
#rm ./models/bonsai/ggml-bonsai-1b-f16.bin

# done!
cd ./scripts/bonsai
echo "Done. Run 'chat-bonsai-1b.sh' to test model."
