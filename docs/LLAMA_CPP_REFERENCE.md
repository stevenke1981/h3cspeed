# How llama.cpp influenced h3cspeed

The useful reference is GGML's backend boundary, not the llama transformer
model itself.

## Adopted concepts

- C ABI between model code and backend implementation;
- explicit backend device and buffer ownership;
- asynchronous host/device transfers;
- capability probing instead of product-name checks;
- backend-specific code isolated from model parsing and CLI behavior;
- room for CPU, CUDA and future backends behind one interface.

## Deliberately not copied

- no llama model loader;
- no GGUF conversion requirement;
- no dependency on llama.cpp or GGML in v0.1;
- no attempt to express every H3 Audio/Video VAE operation as a generic llama
  graph node;
- no source files copied from llama.cpp.

This choice keeps the original H3 safetensor checkpoint usable and makes the
port reviewable: changes are limited to the existing accelerator seam and two
Apple host dependencies.
