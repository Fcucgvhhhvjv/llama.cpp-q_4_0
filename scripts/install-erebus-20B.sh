#!/bin/bash

# cd to scripts dir
cd `dirname $0`

# download model to models dir
echo "Downloading model"
python ./convert_gptneox_to_ggml.py KoboldAI/GPT-NeoX-20B-Erebus ../models/erebus

# remove temp cache dir
echo "Removing temp cache dir"
rm -r ../models/erebus-cache

# quantize model
echo "Quantizing model"
cd ..
python ./scripts/quantize-gptneox.py ./models/pythia/ggml-GPT-NeoX-20B-Erebus-f16.bin

# remove non-quantized model
echo "Remove non-quantized model"
rm ./models/pythia/ggml-GPT-NeoX-20B-Erebus-f16.bin

# done!
cd scripts
echo "Done. Run 'chat-erebus-20B.sh' to test model."
