# Third-party notices

## antirez/h3.c

`h3cspeed` downloads a pinned copy of `antirez/h3.c` during bootstrap. Those
files are distributed under their upstream MIT license. The bootstrap process
retains upstream `LICENSE` and `THIRD_PARTY_NOTICES.md` inside
`third_party/h3`.

Pinned revision:

```text
8974cc055ea9c02fcd14cc27dfda3e1027c05153
```

## MiniMax-H3 model snapshot

The optional `MiniMaxAI/MiniMax-H3` download is pinned to revision
`939557dc319dd91227e30195a763f272ba7f8765`. Model weights are not included in
the source archive. This notice does not assert a license for the model;
consult the upstream distribution terms before use.

## yyjson

The portable tokenizer builds with yyjson commit
`9ddba001a4ea88e93b46932e5c5b87b222e19a5f` through CMake FetchContent.
yyjson is distributed under the MIT license.

## ICU

Unicode normalization and category handling use the system ICU C library.
ICU carries the Unicode/ICU license; consult the installed package for its
complete notices.

## llama.cpp / GGML

`llama.cpp` is referenced for backend architecture concepts only. No llama.cpp
or GGML source code is included in this package, and h3cspeed has no runtime or
build dependency on it.
