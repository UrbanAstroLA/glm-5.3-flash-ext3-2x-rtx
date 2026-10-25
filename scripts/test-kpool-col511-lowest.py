#!/usr/bin/env python3
"""CPU regression gate: the kpool indexer discards the lowest-scoring selected pool (port-kpool-col511-lowest-glm53.py).

Checks, on the installed vLLM tree (no GPU, no model, no network):
  - decode: after the swap, column 511 holds each row's lowest-scoring selected pool and the selected set is
    unchanged (unordered top-k output with a few unfilled slots);
  - prefill: a row with exactly 512 candidates, returned in index order, discards its lowest-scoring pool rather
    than the newest one.
The top-k output order is emulated with the order the B12x kernels were measured to return (the kernels need
a GPU): a random permutation for the tiled decode route, index order for 512-candidate prefill rows.

Run inside the recipe image:
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD/scripts:/s:ro" \
    --entrypoint python3 IMAGE /s/test-kpool-col511-lowest.py [--expect-unpatched] [--output report.json]
--expect-unpatched reproduces the omission on v0.9.1; the default checks the port.
"""
import argparse
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
gen = torch.Generator().manual_seed(1)
RESULTS = []


def check(name, holds, detail=""):
    met = bool(holds) == PATCHED
    RESULTS.append(dict(name=name, holds=bool(holds), met=met, detail=detail))
    print(f"{'PASS' if met else 'FAIL'} {name}: {'holds' if holds else 'does not hold'} {detail}")


def column_511_is_lowest(pool_topk, scores_of):
    s = scores_of(pool_topk)
    return bool((s[:, -1] <= s.min(dim=1).values).all())


rows, k = 64, 512
score_table = torch.randn(rows, 4096, generator=gen)
# decode, tiled route: the 512 selected pools come back unordered (emulated as a random permutation),
# with a few unfilled slots (-1, score -inf)
topk = torch.stack([torch.randperm(4096, generator=gen)[:k] for _ in range(rows)])
topk[:4, 100:110] = -1


def scores_of(t):
    return torch.where(t >= 0, score_table.gather(1, t.clamp(min=0)), torch.tensor(float("-inf")))


before = {r: set(topk[r].tolist()) for r in range(rows)}
if hasattr(kp, "_move_lowest_selected_pool_last"):
    kp._move_lowest_selected_pool_last(topk, scores_of(topk))
same_set = all(set(topk[r].tolist()) == before[r] for r in range(rows))
check("col511 decode: column 511 holds each row's lowest-scoring selected pool, set unchanged",
      column_511_is_lowest(topk, scores_of) and same_set)

# prefill: a row with exactly 512 candidates comes back in index order (pool 511 last), ids relative
# to the row start; the slice [:, :511] then drops the newest pool whatever its score
row_start, logits = 7, torch.randn(1, 7 + 512 + 5, generator=gen)
ptopk = torch.arange(512, dtype=torch.int32)[None].clone()
starts = torch.tensor([row_start], dtype=torch.int32)
if hasattr(kp, "_prefill_selected_pool_scores"):
    kp._move_lowest_selected_pool_last(ptopk, kp._prefill_selected_pool_scores(logits, ptopk, starts))
lowest = column_511_is_lowest(ptopk, lambda t: logits.gather(1, t.long() + row_start))
check("col511 prefill: a 512-candidate row discards its lowest-scoring pool, not pool 511", lowest,
      f"(column 511 holds pool {int(ptopk[0, -1])})")

ok = all(r["met"] for r in RESULTS)
print(f"{'ALL MET' if ok else 'NOT MET'}: {sum(r['met'] for r in RESULTS)}/{len(RESULTS)} "
      f"({'port expected' if PATCHED else 'v0.9.1 omissions expected'})")
if args.output:
    json.dump(dict(expect="patched" if PATCHED else "unpatched", results=RESULTS), open(args.output, "w"), indent=1)
raise SystemExit(0 if ok else 1)
