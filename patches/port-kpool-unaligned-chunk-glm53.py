#!/usr/bin/env python3
"""Write index-K pools that straddle an unaligned prefill chunk start.

REQUIRES port-kpool-tail-positions-glm53.py, applied first. The repair reads a
straddling pool's earlier keys from the request's own tail ring. Only with that
port is the ring addressed by token position and owned by the request
(``positions`` reach the tail metadata builder, null-block rows get PAD, and the
circular mapping is graph-safe); without it most tail writes go to the null
block, so the ring would not hold the request's keys and the repair would read
foreign or stale rows. This script checks for every edit of that port and
refuses, writing nothing, if any is missing.

``_kpool_compress_insert`` pools batch rows t-3..t and writes a pool only when
row t completes it; its docstring assumes pool-aligned chunk starts. The
scheduler does not align prefill chunks to 4 tokens: with DFlash the per-request
input budget is reduced by the draft slots (2,048 - 3 = 2,045), each decode row
in the step takes 1 + K tokens plus K draft slots, and requests sharing a step
split the remainder. At an unaligned start the pool straddling it is never
written when the request is the step's first prefill row, and is compressed from
the previous request's last tokens when it is a later row. A later read of that
pool sees uninitialised or foreign keys.

A pool member that is not one of the same request's rows in this batch (same tail
block, position pos - j) is now read from the request's tail ring at pos % ring,
where its earlier chunk seeded it; the ring is read before this step's seed
overwrites it. Rows without a tail slot, and calls without the ring, keep the
previous rule. No host sync. SGLang handles the same case writer-side
(kpool_plan.py ``_decompose_compress``, ``n_from_tail``). Requires
port-kpool-tail-positions-glm53.py.
Usage: port-kpool-unaligned-chunk-glm53.py VLLM_DIR
"""

import os
import sys
import tempfile
from pathlib import Path

vllm = Path(sys.argv[1])
PENDING: list[tuple[Path, str]] = []


def patch(rel: str, edits: list[tuple[str, str, int]], marker: str) -> None:
    """Validate every edit before anything is written (see the end of the file)."""
    path = vllm / rel
    source = path.read_text()
    if marker in source:
        print(f"{rel}: already applied")
        return
    for old, new, count in edits:
        if source.count(old) != count:
            raise SystemExit(f"{rel}: refusing, source drift ({source.count(old)} != {count}):{chr(10)}{old}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    PENDING.append((path, source))


# Prerequisite: port-kpool-tail-positions-glm53.py. The repair reads the request's
# tail ring by position, which is correct only with that port's circular,
# request-owned, graph-safe mapping. Check each of its edits; refuse otherwise.
REQUIRED = [
    ("v1/worker/gpu/model_states/mamba_hybrid.py", "positions=input_batch.positions,"),
    ("v1/attention/backends/mla/indexer.py", "torch.where(own_block != 0, slots, -1)"),
    ("v1/attention/backends/mla/indexer.py", "common_attn_metadata.slot_mapping.copy_(slot_mapping)"),
]
for rel, needle in REQUIRED:
    if needle not in (vllm / rel).read_text():
        raise SystemExit(
            "kpool-unaligned-chunk: refusing, nothing written: this port requires "
            f"port-kpool-tail-positions-glm53.py (missing in {rel}); apply it first"
        )


patch('model_executor/layers/sparse_attn_indexer_kpool.py', [
    (
'''    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.
''',
'''    kpool: int,
    head_dim: int,
    round_scale: bool,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_slot_mapping: torch.Tensor | None = None,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.
''',
        1,
    ),
    (
'''    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first (``torch.nonzero`` + boolean-mask gather + an
    ``ok.all()`` check) costs two device syncs on the eager prefill path and
    buys nothing numerically. Assumes pool-aligned chunk starts (same
    invariant as sglang).
    """
    n = slot_mapping.shape[0]
    # No pool can complete in a batch smaller than one pool; also keeps the
    # clamped gather indices below in bounds.
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
''',
'''    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first (``torch.nonzero`` + boolean-mask gather + an
    ``ok.all()`` check) costs two device syncs on the eager prefill path and
    buys nothing numerically.

    kpool-unaligned-chunk: the scheduler does not start prefill chunks on pool boundaries
    (budget minus draft slots, shared-step splits). With ``positions``, the
    tail ring and its token-granular slots, a pool member that is not one of
    the same request's tokens in this batch (same tail block, position
    ``pos - j``) is read from the request's tail ring at ``pos % ring``, where
    its earlier chunk seeded it (kpool_seed_tail_cache). Rows without a tail
    slot, or calls without the ring, keep the pool-aligned rule.
    """
    n = slot_mapping.shape[0]
    repair = (
        positions is not None
        and tail_kv_cache is not None
        and tail_slot_mapping is not None
    )
    # No pool can complete in a batch smaller than one pool (unless its earlier
    # members come from the tail ring); the gather indices are clamped below.
    if n == 0 or (n < kpool and not repair):
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
''',
        1,
    ),
    (
"""    write_mask = valid & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    kpool_compress_and_write_cache(
        kv_cache,
        k[idx],  # [n, kpool, head_dim]
        gate_score[idx],
        ape,
        slot_mapping.to(torch.int64),
        pool_size=kpool,
""",
"""    write_mask = valid & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    idx = idx.clamp_max(n - 1)  # no-op when n >= kpool
    if not repair:
        slot_k, slot_score = k[idx], gate_score[idx]
    else:
        ring = tail_kv_cache.shape[2]
        tpos = positions[:n].to(torch.int64)
        tslot = tail_slot_mapping[:n].to(torch.int64)
        has_ring = tslot >= 0
        blk = torch.div(tslot.clamp_min(0), ring, rounding_mode="floor")
        mpos = tpos[:, None] - (kpool - 1) + offs[None, :]  # member positions
        rel = pos[:, None] - (kpool - 1) + offs[None, :]  # member batch rows
        relc = rel.clamp(0, n - 1)
        own = (
            (rel >= 0)
            & has_ring[relc]
            & (blk[relc] == blk[:, None])
            & (tpos[relc] == mpos)
        )
        from_ring = has_ring[:, None] & ~own
        src = torch.where(has_ring[:, None], relc, idx)
        slot_k, slot_score = k[src], gate_score[src]
        rslot = torch.remainder(mpos, ring)
        rblk = blk[:, None].expand(-1, kpool)
        sel = from_ring[..., None]
        slot_k = torch.where(sel, tail_kv_cache[rblk, 0, rslot], slot_k)
        slot_score = torch.where(sel, tail_kv_cache[rblk, 1, rslot], slot_score)
        write_mask = torch.where(
            has_ring, valid & (own | (mpos >= 0)).all(dim=1), write_mask
        )
    kpool_compress_and_write_cache(
        kv_cache,
        slot_k,  # [n, kpool, head_dim]
        slot_score,
        ape,
        slot_mapping.to(torch.int64),
        pool_size=kpool,
""",
        1,
    ),
    (
"""            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
""",
"""            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                # kpool-unaligned-chunk: a pool straddling an unaligned chunk start takes
                # its earlier members from the request's tail ring.
                d7_tail = None
                if (
                    tail_kv_cache is not None
                    and tail_prefix is not None
                    and positions is not None
                    and os.environ.get("VLLM_KPOOL_SKIP_TAIL_CACHE") != "1"
                ):
                    d7_tail = attn_metadata.get(_resolve_layer_name(tail_prefix))
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
""",
        1,
    ),
    (
"""                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
                # Persist the prefill tail (trailing incomplete pool's raw K +
                # gate score) into the paged tail cache, so the decode side can
""",
"""                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                    positions=None if d7_tail is None else positions[prefill_slice],
                    tail_kv_cache=None if d7_tail is None else tail_kv_cache,
                    tail_slot_mapping=(
                        None if d7_tail is None else d7_tail.slot_mapping[prefill_slice]
                    ),
                )
                # Persist the prefill tail (trailing incomplete pool's raw K +
                # gate score) into the paged tail cache, so the decode side can
""",
        1,
    ),
], marker='kpool-unaligned-chunk')



for path, source in PENDING:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".port-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(source)
        os.chmod(tmp, os.stat(path).st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

print('GLM kpool prefill writer repairs pools straddling unaligned chunk starts (kpool-unaligned-chunk)')
