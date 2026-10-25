#!/usr/bin/env python3
"""CPU regression gate: the kpool tail ring is addressed by position (port-kpool-tail-positions-glm53.py).

Checks, on the installed vLLM tree (no GPU, no model, no network):
  - MambaHybridModelState passes positions to build_attn_metadata;
  - rows on the null block (padding and dummy requests) get PAD, never block 0's ring;
  - KpoolTailMetadataBuilder writes the circular mapping into the persistent slot-mapping buffer
    (the address a FULL CUDA graph replay reads).

Run inside the recipe image:
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD/scripts:/s:ro" \
    --entrypoint python3 IMAGE /s/test-kpool-tail-positions.py [--expect-unpatched] [--output report.json]
--expect-unpatched reproduces each omission on v0.9.1; the default checks the port.
"""
import argparse
import ast
import inspect
import json
from types import SimpleNamespace as NS

import torch

torch.Tensor.pin_memory = lambda self, *a, **k: self  # CPU: no pinned host allocator

import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401  (import order of the engine)
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
import vllm.v1.attention.backends.mla.indexer as idx  # noqa: E402
import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
ap.add_argument("--expect-unpatched", action="store_true")
ap.add_argument("--output", default="")
args = ap.parse_args()
PATCHED = not args.expect_unpatched
RING = 8
RESULTS = []


def check(name, holds, detail=""):
    met = bool(holds) == PATCHED
    RESULTS.append(dict(name=name, holds=bool(holds), met=met, detail=detail))
    print(f"{'PASS' if met else 'FAIL'} {name}: {'holds' if holds else 'does not hold'} {detail}")


def kwargs_of_calls(source, func_name):
    """Keyword names of every call to func_name (attribute or plain name) in source."""
    tree = ast.parse(source)
    return [{k.arg for k in n.keywords} for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", getattr(n.func, "id", None)) == func_name]


def tail_cm(per_req, own_blocks, pad=4):
    positions = torch.cat([torch.tensor(p, dtype=torch.int64) for p in per_req])
    qsl = torch.zeros(len(per_req) + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor([len(p) for p in per_req]), 0, out=qsl[1:])
    bt = torch.zeros(len(per_req), 64, dtype=torch.int32)
    bt[:, 0] = torch.tensor(own_blocks, dtype=torch.int32)
    seq = torch.tensor([max(p) + 1 for p in per_req], dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=qsl, query_start_loc_cpu=qsl.clone(), seq_lens=seq, num_reqs=len(per_req),
        num_actual_tokens=positions.numel(), max_query_len=max(map(len, per_req)), max_seq_len=int(seq.max()),
        block_table_tensor=bt, slot_mapping=torch.full((positions.numel() + pad,), -1, dtype=torch.int64),
        positions=positions)


kws = kwargs_of_calls(inspect.getsource(mh.MambaHybridModelState), "build_attn_metadata")
check("tail-positions: MambaHybridModelState passes positions to build_attn_metadata",
      any("positions" in k for k in kws))

cm = tail_cm([list(range(10)), [0, 1]], [5, 0])  # second request sits on the null block (padding/dummy)
out = idx.compute_kpool_tail_slot_mapping(cm.slot_mapping, cm.block_table_tensor, cm.query_start_loc,
                                          cm.positions, cm.num_actual_tokens, cm.num_reqs, RING)
check("tail-positions: rows on the null block get PAD, never block 0's ring",
      out[10:12].tolist() == [-1, -1], f"(null-block rows -> {out[10:12].tolist()})")

builder = object.__new__(idx.KpoolTailMetadataBuilder)
builder.kv_cache_spec = NS(block_size=RING)
cm = tail_cm([list(range(10)), list(range(12))], [5, 9])
ptr = cm.slot_mapping.data_ptr()
md = idx.KpoolTailMetadataBuilder.build(builder, 0, cm)
circular = all(int(md.slot_mapping[i]) // RING == (5 if i < 10 else 9) for i in range(22))
check("tail-positions: the circular mapping is written into the persistent slot-mapping buffer",
      circular and md.slot_mapping.data_ptr() == ptr, f"(same storage: {md.slot_mapping.data_ptr() == ptr})")

ok = all(r["met"] for r in RESULTS)
print(f"{'ALL MET' if ok else 'NOT MET'}: {sum(r['met'] for r in RESULTS)}/{len(RESULTS)} "
      f"({'port expected' if PATCHED else 'v0.9.1 omissions expected'})")
if args.output:
    json.dump(dict(expect="patched" if PATCHED else "unpatched", results=RESULTS), open(args.output, "w"), indent=1)
raise SystemExit(0 if ok else 1)
