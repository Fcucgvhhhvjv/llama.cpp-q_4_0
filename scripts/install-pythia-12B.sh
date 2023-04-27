#!/bin/bash

# cd to scripts dir
cd `dirname $0`

# download model to models dir
echo "Downloading model"
python ./convert_gptneox_to_ggml.py OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 ../models/pythia

# remove temp cache dir
echo "Removing temp cache dir"
rm -r ../models/pythia-cache

# quantize model
echo "Quantizing model"
cd ..
python ./quantize-gptneox.py ../models/pythia/ggml-oasst-sft-4-pythia-12b-epoch-3.5-f16.bin

# remove non-quantized model
echo "Remove non-quantized model"
rm ./models/pythia/ggml-oasst-sft-4-pythia-12b-epoch-3.5-f16.bin

# done!
cd scripts
echo "Done. Run 'chat-pythia-12B.sh' to test model."
