#!/bin/bash

# Install eninops if not installed
pip install einops

# cd to scripts dir
cd `dirname $0`

# download model to models dir
echo "Downloading model"
python ./convert-falcon-to-ggml.py tiiuae/falcon-7b-instruct ../../models/falcon

# remove temp cache dir
echo "Removing temp cache dir"
#rm -r ../../models/falcon-cache

# quantize model
echo "Quantizing model"
cd ../..
python ./scripts/falcon/quantize-falcon.py ./models/falcon/ggml-falcon-7b-instruct-f16.bin

# remove non-quantized model
echo "Remove non-quantized model"
#rm ./models/falcon/ggml-falcon-7b-instruct-f16.bin

# done!
cd ./scripts/falcon
echo "Done. Run 'chat-falcon-7B-intruct.sh' to test model."
