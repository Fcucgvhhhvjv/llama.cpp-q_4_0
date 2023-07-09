# llm.cpp

A simple demonstration on how to adapt ggerganov/llama.cpp to use other large language models.
- OpenAssistant (GPT-NeoX based) models like stablelm and oasst.
- RWKV (v4)
- RefinedWeb

## Notes:
- All gptneox source files are in the examples/gptneox directory, minimal changes have been made to maybe merge with llama.cpp in the future. Only the makefile has been modified. GPT-NeoX related scripts are all located in the scripts directory.
- This version does NOT separate the attention weights (QKV to Q, K, V) so that applying LoRA adapters from huggingface will be simpler. The model structure is kept as close to the huggingface version as possible when converting to ggml format.
- The gptneox chat scripts use main-oasst, which is different than the llama main, some parameters are the same but it is written to be used with oasst specifically. You can use main-gptneox instead if you like, which is essentially a carbon copy of llama main. If you choose to use main-gptneox then be sure to treat the parameters similar to alpaca chat. I have not spent too much time using these command line mains because I wrote my own MacOS catalyst app that interacts with a gptneox.cpp as a swift package. They are only there for testing or sanity check.
- The model installer scripts will automatically download from huggingface, convert to ggml, and then quantize to "q4_0". Intermediate caches and f16 models will be removed during this process. Feel free to comment the "rm" commands in the script if you want to keep these temp intermediate models.
- Chances are I will expand on this more as OpenAssistant progresses. Would be nice to have this merged with ggml/llama at some point so the community can take the wheel. Just wanted to give you guys something right now to work with.

## Directions:

To install models:

- StableLM 7B model:
    run ./scripts/install-stablelm-7B.sh
    This will install a "q4_0" quantized model to ./models/stablelm
    
- Pythia 12B model - run ./scripts/install-pythia-12B.sh
    This will install a "q4_0" quantized model to ./models/pythia
    
To run models:

- StableLM 7B model:
    run ./scripts/chat-stablelm-7B.sh

- Pythia 12B model:
    run ./scripts/chat-pythia-12B.sh
    
Pre-converted/quantized model weights are available via Huggingface:
- StableLM 7B https://huggingface.co/byroneverson/ggml-stablelm-7b-sft-v7-epoch-3-q4_0
- Pythia 12B https://huggingface.co/byroneverson/ggml-oasst-sft-4-pythia-12b-epoch-3.5-q4_0
