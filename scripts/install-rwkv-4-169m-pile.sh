#!/bin/bash

# cd to scripts dir
cd `dirname $0`

# download model to models dir
echo "Downloading model"
python ./convert-rwkv-to-ggml.py RWKV/rwkv-4-169m-pile ../models/rwkv # true

# remove temp cache dir
echo "Removing temp cache dir"
#rm -r ../models/rwkv-cache

# quantize model
echo "Quantizing model"
cd ..
python ./scripts/quantize-rwkv.py ./models/rwkv/ggml-rwkv-4-169m-pile-f16.bin

# remove non-quantized model
echo "Remove non-quantized model"
#rm ./models/rwkv/ggml-rwkv-7b-pile-f16.bin

# done!
cd scripts
echo "Done."
