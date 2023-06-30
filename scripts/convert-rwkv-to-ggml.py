# Convert Hugging Face RWKV models to ggml format

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: python convert-hf-to-ggml.py model_name dir-output [use-f32]")
    print("  model_name: name of the model to convert. Example: 'bigscience/bloomz-560m'")
    print("  dir-output: directory where the output file will be written")
    print("  use-f32:    if present, use float32 instead of float16")
    sys.exit(1)

model_name = sys.argv[1]
dir_out = sys.argv[2]
model_cache_dir = dir_out + "-cache"

# make sure the output directory exists
os.makedirs(dir_out, exist_ok=True)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]
ftype = 1
if len(sys.argv) > 3:
    ftype = 0

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading model: ", model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if ftype == 1 else torch.float32, cache_dir=model_cache_dir)
model.eval()
for p in model.parameters():
    p.requires_grad = False
hparams = model.config.to_dict()
print("Model loaded: ", model_name)

print(hparams)

fn_bin = f"/ggml-{model_name.split('/')[-1]}-{ftype_str[ftype]}.bin"
fn_out = dir_out + fn_bin
fout = open(fn_out, "wb")

# 0x67676d6c is unversioned ggml
# 0x67676d66 is versioned ggmf (requires token scores)
ggml_file_magic = 0x67676d6c
#ggml_file_version = 0x00000001 # v1

hparams["multiple_of"] = 1
fout.write(struct.pack("i", ggml_file_magic)) # magic: ggml in hex
#fout.write(struct.pack("i", ggml_file_version))
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["context_length"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
#fout.write(struct.pack("i", hparams["num_attention_heads"]))
fout.write(struct.pack("i", hparams["num_hidden_layers"]))
fout.write(struct.pack("i", hparams["rescale_every"]))
#fout.write(struct.pack("i", int((hparams["hidden_size"] / hparams["num_attention_heads"]) * hparams["rotary_pct"]))) # rotary_dim
#fout.write(struct.pack("i", int(hparams["use_parallel_residual"])))
fout.write(struct.pack("i", ftype))

# Is this correct??
dot_token = tokenizer.encode(".")[0]
for i in range(hparams["vocab_size"]):
    text = tokenizer.decode([i]).encode('utf-8')
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

# Write tensors
list_vars = model.state_dict()
rescale_every = hparams["rescale_every"]
print(f'Writing/converting {len(list_vars.keys())} tensors.')
for (index, name) in enumerate(list_vars.keys()):
    # No gradients for these
    list_vars[name].requires_grad = False
    
    # Current tensor
    tensor = list_vars[name].float()
    
    # Remove dimensions with only a single element (1, 1, m, n) -> (m, n)
    if '.time_' in name:
        tensor = tensor.squeeze()
    
    # Adjust time_decay, probably for the best once the models are working, can also be performed during inference
    if '.time_decay' in name:
        tensor = -torch.exp(tensor)
    
    # Rescale layers if enabled, only attn output w or ff value weight
    rescaled_tensor = 'f'
    if rescale_every > 0:
        if ('.attention.output.weight' in name) or ('.feed_forward.value.weight' in name):
            block_id = int(name.split('.')[2])
            tensor = torch.div(tensor, 2 ** int(block_id // rescale_every))
            rescaled_tensor = 't'
    
    # Shape of tensor
    shape = tensor.shape
    
    # Keep 1-dim vectors in float32, otherwise float16
    if ftype == 1 and len(shape) > 1:
        tensor = tensor.half()
        
    print(f'{index: >4} {name: >50} {tensor.dtype} rescaled={rescaled_tensor} {shape}')
    
    # Header
    name_encoded = name.encode('utf-8')
    #fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    #for i in range(n_dims):
    #    fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    #print(str)
    
    # Write len of shape, len of name, and float type
    fout.write(struct.pack(
        '=iii',
        len(shape),
        len(name_encoded),
        1 if tensor.dtype == torch.float16 else 0
    ))
    # Dimension order is reversed here:
    # * PyTorch shape is (x rows, y columns)
    # * ggml shape is (y elements in a row, x elements in a column)
    # Both shapes represent the same tensor.
    for dim in reversed(tensor.shape):
        fout.write(struct.pack('=i', dim))
    # Write name
    fout.write(name_encoded)
    
    # Get tensor data as float32
    #data = tensor.numpy()
    #data = data.astype(np.float32)

    # Get # of dims, 1 or 2
    #n_dims = len(data.shape)
    #print(name, n_dims, data.shape)

    # Get float type, default is fp32
    #ftype_cur = 0
    #if ftype == 1 and n_dims > 1:
    #    print("  Converting to float16", data.shape, data[:3, :3].tolist())
    #    data = data.astype(np.float16)
    #    ftype_cur = 1
    #else:
    #    print("  Converting to float32", data.shape,
    #          data[:3, :3].tolist() if n_dims > 1 else data[:3].tolist())
    #    data = data.astype(np.float32)
    
    # Write data to file
    tensor.numpy().tofile(fout)

fout.close()

print("Done. Output file: " + fn_out)
print("")

