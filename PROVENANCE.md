# Source provenance

## MiniMax-H3 engine base

- Repository: `antirez/h3.c`
- Pinned commit: `8974cc055ea9c02fcd14cc27dfda3e1027c05153`
- License: MIT
- Integration: downloaded by `scripts/bootstrap.py`, verified with pinned Git
  blob IDs, then patched narrowly for the portable scaler and CLI name.

The v0.2 RAM/file offload implementation is an h3cspeed overlay change. It does
not modify safetensor values or tensor layout; it changes where read-only tensor
bytes reside between operations.

## MiniMax-H3 model snapshot

- Repository: `MiniMaxAI/MiniMax-H3`
- Pinned revision: `939557dc319dd91227e30195a763f272ba7f8765`
- Use: optional external download performed by `scripts/download_h3_fl2va.py`;
  model weights are not redistributed in this source repository.
- Licensing: this project does not assert a model license. Review the upstream
  model distribution terms before downloading or using the snapshot.

## llama.cpp / GGML reference

- Repository reviewed: `ggml-org/llama.cpp`
- Reference commit:
  `f785fc9ea485e6cfdda129978310aa52939c3619`
- Use: architectural reference for C ABI stability, backend-owned buffers,
  devices, streams/events, capability-driven dispatch and bounded accelerator
  residency.
- No `llama.cpp` or GGML source is copied into this archive and it is not a
  build/runtime dependency.

## NVIDIA design inputs

The memory policy follows documented CUDA distinctions between explicit device
allocation, pinned host allocation and managed-memory behavior. The project
uses explicit `cudaMalloc`, host backing, streams and events; it does not use
managed-memory oversubscription.

## Overlay dependencies

- CUDA Runtime, cuBLAS and cuBLASLt from the locally installed CUDA Toolkit.
- ICU (`icu-uc`) from the host system.
- yyjson commit `9ddba001a4ea88e93b46932e5c5b87b222e19a5f` fetched by CMake.
- FFmpeg/FFprobe invoked by the retained upstream media layer.
