#!/usr/bin/env python3
"""CPU regression gate: index-K pools straddling an unaligned prefill chunk start (port-kpool-unaligned-chunk-glm53.py).

Requires port-kpool-tail-positions-glm53.py in the tree under test, as the port itself does.

Runs the installed _kpool_compress_insert on one prefill step whose request starts a chunk at positions
2045, 2046, 2047 and 4090 (residues 1, 2, 3 and 2 mod 4), once as the step's first prefill row and once behind
another request's chunk. The compress kernel is replaced by a recorder that identifies every pooled member by
its exact key and gate bytes, so the check compares token identities, not numerics. Each request's preceding
tokens are placed in its tail ring at pos % ring, as the prefill tail seed leaves them. Also checks that the
prefill call site passes the tail ring to the writer.

Run inside the recipe image:
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD/scripts:/s:ro" \
    --entrypoint python3 IMAGE /s/test-kpool-unaligned-chunk.py [--expect-unpatched] [--output report.json]
--expect-unpatched reproduces the omission on v0.9.1; the default checks the port.
"""
import argparse
import ast
import inspect
import json

import torch

torch.Tensor.pin_memory = lambda self, *a, **k: self  # CPU: no pinned host allocator

import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401  (import order of the engine)
import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: E402,F401
from vllm.model_executor.layers import sparse_attn_indexer_kpool as kp  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
ap.add_argument("--expect-unpatched", action="store_true")
ap.add_argument("--output", default="")
args = ap.parse_args()
PATCHED = not args.expect_unpatched
KPOOL, HD, RING = 4, 128, 8
gen = torch.Generator().manual_seed(1)
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


def raw(t):
    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()


has_ring = "tail_slot_mapping" in inspect.signature(kp._kpool_compress_insert).parameters
ape = (torch.randn(KPOOL, HD, generator=gen) * 0.1).float()
worst = []
for start in (2045, 2046, 2047, 4090):
    for role in ("first", "later"):
        # name: (keys, gates, index-K block, tail block)
        reqs = {"A": (torch.randn(start + 16, HD, generator=gen).to(torch.bfloat16),
                      torch.randn(start + 16, HD, generator=gen).to(torch.bfloat16), 3, 2),
                "P": (torch.randn(128, HD, generator=gen).to(torch.bfloat16),
                      torch.randn(128, HD, generator=gen).to(torch.bfloat16), 5, 3)}
        rows = ([] if role == "first" else [("P", p) for p in range(100, 112)]) + \
            [("A", p) for p in range(start, start + 12)]
        table = {(raw(kk[p]), raw(gg[p])): (n, p) for n, (kk, gg, _, _) in reqs.items() for p in range(len(kk))}

        def loc(name, pool):
            return reqs[name][2] * 1088 + pool % 1088

        written = {}

        def recorder(kv_cache, slot_k, slot_score, ape_, slots, pool_size, head_dim=HD, write_mask=None, **_):
            for r in range(slot_k.shape[0]):
                if int(slots[r]) >= 0 and (write_mask is None or bool(write_mask[r])):
                    written[int(slots[r])] = [table.get((raw(slot_k[r, j]), raw(slot_score[r, j])), ("?", -1))
                                              for j in range(pool_size)]

        k = torch.stack([reqs[n][0][p] for n, p in rows])
        g = torch.stack([reqs[n][1][p] for n, p in rows])
        sm = torch.tensor([loc(n, p // KPOOL) if (p + 1) % KPOOL == 0 else -1 for n, p in rows])
        extra = {}
        if has_ring:  # each request's preceding tokens sit in its tail ring at pos % ring (prefill tail seed)
            tail = torch.zeros(4, 2, RING, HD, dtype=torch.bfloat16)
            for n, first in (("A", start), ("P", 100)):
                kk, gg, _, tb = reqs[n]
                for p in range(first - RING, first):
                    tail[tb, 0, p % RING], tail[tb, 1, p % RING] = kk[p], gg[p]
            extra = dict(positions=torch.tensor([p for _, p in rows]), tail_kv_cache=tail,
                         tail_slot_mapping=torch.tensor([reqs[n][3] * RING + p % RING for n, p in rows]))
        saved = kp.kpool_compress_and_write_cache
        kp.kpool_compress_and_write_cache = recorder
        try:
            kp._kpool_compress_insert(k, g, ape, torch.empty(0, dtype=torch.uint8), sm, KPOOL, HD, True, **extra)
        finally:
            kp.kpool_compress_and_write_cache = saved
        for p in range(start, start + 12):
            if (p + 1) % KPOOL == 0:
                pool = p // KPOOL
                want = [("A", q) for q in range(pool * KPOOL, pool * KPOOL + KPOOL)]
                got = written.get(loc("A", pool))
                if got != want:
                    worst.append(f"start {start} ({role}) pool {pool}: "
                                 f"{'unwritten' if got is None else ' '.join(f'{n}{q}' for n, q in got)}")
check("unaligned-chunk: every pool completed in the chunk holds the request's own four tokens",
      not worst, "(" + ("; ".join(worst[:3]) + (" ..." if len(worst) > 3 else "")) + ")" if worst else "")
call_kws = kwargs_of_calls(inspect.getsource(kp.sparse_attn_indexer_kpool), "_kpool_compress_insert")
check("unaligned-chunk: the prefill call site passes the tail ring to the writer",
      any("tail_slot_mapping" in k for k in call_kws))

ok = all(r["met"] for r in RESULTS)
print(f"{'ALL MET' if ok else 'NOT MET'}: {sum(r['met'] for r in RESULTS)}/{len(RESULTS)} "
      f"({'port expected' if PATCHED else 'v0.9.1 omissions expected'})")
if args.output:
    json.dump(dict(expect="patched" if PATCHED else "unpatched", results=RESULTS), open(args.output, "w"), indent=1)
raise SystemExit(0 if ok else 1)
