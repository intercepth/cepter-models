# Cepter EFX – GPU models for Voice Isolation

These are the models the **GPU mode of Voice Isolation** in the Cepter EFX After Effects
extension downloads. They are re-exports of the MIT-licensed StemSplit ONNX conversions of
Meta's HT-Demucs, changed so that they run correctly and within a normal amount of memory
on DirectML (tested on an NVIDIA GPU and an AMD integrated GPU).

| File | Stems | Size | SHA-256 |
|---|---|---|---|
| `cepter_htdemucs_dml.onnx` | 4 (drums, bass, other, vocals) | 169.4 MB | `E5A9C63911266AE18134722994BB503763EC682679B4DF39515A5088CFC4547A` |
| `cepter_htdemucs_6s_dml.onnx` | 6 (adds guitar, piano) | 139.7 MB | `77AD71401CFEBE66D2C304C144B3189A61FF27EFB2A64AE7A2CC5DB1999285FA` |

Same input and output contract as the originals: input `mix` `float32 [1, 2, 343980]`
(7.8 s of 44.1 kHz stereo), output `stems` `float32 [1, S, 2, 343980]`. The weights are
unchanged (stored as fp16 and cast to fp32 at load, losslessly).

## What was changed, and why

The stock models fail on DirectML. Two operations were rewritten into mathematically
identical forms:

1. **The iSTFT `ConvTranspose`** (kernel 4096, stride 1024). On its own it makes DirectML ask
   Windows for tens of GB of *system* memory and fail with `8007000E` (this is not a video
   memory problem – VRAM use stays under 2 GB). It is now a `MatMul` followed by a four-way
   overlap-add.
2. **`Split` feeding the GLU gate** (`Mul(Split_0, Sigmoid(Split_1))`, 48 times). DirectML
   computed this wrongly. Each `Split` is now two `Slice` operations.

The graph was also constant-folded from 24,917 nodes down to about 1,450.

Output was checked against the original model on CPU: maximum difference about 5e-5 on a
signal peaking at 0.35 (4-stem), 2e-5 (6-stem). Measured on an RTX 5050 laptop GPU, a
40-second clip separates in about 4 seconds.

## Licence and attribution

MIT, the same licence as the models these are derived from.

- Original model: HT-Demucs by Rouard, Massa and Défossez, *Hybrid Transformers for Music
  Source Separation* (ICASSP 2023), code and weights at
  <https://github.com/facebookresearch/demucs> (MIT).
- ONNX export and parity verification: StemSplit,
  <https://huggingface.co/StemSplitio/htdemucs-onnx> and
  <https://huggingface.co/StemSplitio/htdemucs-6s-onnx> (MIT).
- DirectML-compatibility changes: Cepter EFX / InterCepth.

Not affiliated with Meta, StemSplit or Microsoft.

```bibtex
@inproceedings{rouard2022hybrid,
  title     = {Hybrid Transformers for Music Source Separation},
  author    = {Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle = {ICASSP 23},
  year      = {2023}
}
```

The files are attached to this repository's **Releases**, not stored in the source tree.
