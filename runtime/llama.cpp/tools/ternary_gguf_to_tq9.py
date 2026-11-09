#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-pack a trained ternary model into TQ9_128, so it runs on the 0005/0006 aarch64 CPU kernels.

Inputs are two GGUF files of the same model:
  F16     the full-precision export (weights already ternary times a scale, e.g. Ternary-Bonsai-1.7B-F16.gguf)
  LIKE    a ternary GGUF of it (e.g. Ternary-Bonsai-1.7B-Q2_0_g64.gguf): every tensor stored there as Q2_0, TQ1_0 or
          TQ2_0 is re-packed from F16 as TQ9_128 with three levels; every other tensor and all metadata are copied from LIKE.
Per 128 weights of a row: s = fp16(max |w|), q = round(w / s) in {-1, 0, +1}. TQ9_128 takes 3.375 bits per weight,
so the file is larger than a 2-bit ternary GGUF (Ternary Bonsai 1.7B: 0.73 GB vs 0.46-0.49 GB).
The script reports how many tensors are reproduced exactly and the largest error relative to each tensor's max |w|;
check the result with llama-perplexity --kl-divergence against the F16 file.

usage: python3 ternary_gguf_to_tq9.py <llama.cpp-tr>/gguf-py F16.gguf LIKE.gguf OUT.gguf
  (gguf-py from the patched tree made by ../apply.sh; it carries the TQ9_128 packer; needs numpy and PyYAML)"""
import sys
import numpy as np

if len(sys.argv) != 5:
    sys.exit(__doc__)
sys.path.insert(0, sys.argv[1])
import gguf
from gguf.quants import TQ9_128

f16, like, out = sys.argv[2:5]
G = 128
TERNARY = {"Q2_0", "TQ1_0", "TQ2_0"}
src = {t.name: t for t in gguf.GGUFReader(f16).tensors}
R = gguf.GGUFReader(like)
W = gguf.GGUFWriter(out, R.fields[gguf.Keys.General.ARCHITECTURE].contents())
for f in R.fields.values():
    if f.name in (gguf.Keys.General.ARCHITECTURE, gguf.Keys.General.FILE_TYPE) or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    W.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)

n = n_exact = 0
worst = 0.0
for t in R.tensors:
    if t.tensor_type.name in TERNARY:
        a = np.asarray(src[t.name].data).astype(np.float32)
        rows, cols = a.shape
        if cols % G:
            sys.exit(f"{t.name}: {cols} columns, not a multiple of {G}")
        xg = a.reshape(rows, cols // G, G)
        s = np.abs(xg).max(-1).astype(np.float16)
        sf = s.astype(np.float32)
        sf[sf == 0] = 1.0
        q = np.clip(np.rint(xg / sf[..., None]), -1, 1).astype(np.int32)
        blk = TQ9_128.pack_codes(s.reshape(-1), q.reshape(-1, G))
        back = gguf.quants.dequantize(blk.reshape(rows, -1), TQ9_128.qtype).reshape(rows, cols)
        worst = max(worst, float(np.abs(back - a).max() / max(float(np.abs(a).max()), 1e-30)))
        n_exact += int(np.array_equal(back, a))
        n += 1
        raw = blk.reshape(rows, -1)
        W.add_tensor(t.name, raw, raw_shape=raw.shape, raw_dtype=TQ9_128.qtype)
    else:
        a = np.asarray(t.data)
        plain = t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        W.add_tensor(t.name, a, raw_shape=None if plain else a.shape, raw_dtype=t.tensor_type)
W.add_uint32(gguf.Keys.General.FILE_TYPE, 43)  # LLAMA_FTYPE_MOSTLY_TQ9_128 in this series
W.write_header_to_file()
W.write_kv_data_to_file()
W.write_tensors_to_file()
W.close()
print(f"{n} tensors re-packed as TQ9_128 ({n_exact} reproduce the F16 values exactly; "
      f"largest error {worst:.2e} of a tensor's max |w|) -> {out}")
