#!/usr/bin/env python3
"""Discard the lowest-scoring selected pool, not column 511 (kpool indexer).

The indexer asks the top-k for ``index_topk // index_kpool`` (512) pools and
keeps columns 0..510, so that 511 pools plus the newest 1-3 tail tokens fit the
2,048-wide token index. No top-k route sorts its output, so the discarded pool is
whatever the kernel left in column 511. On the B12x tiled decode route (more
than 16 query rows) that pool ranked better than 480 of 512 in most live decode
rows; prefill rows with exactly 512 candidates come back in index order, so the
newest complete pool was discarded. The selected set itself was correct.

Decode now requests the selected scores on every DCP size and swaps each row's
lowest-scoring selected pool into the last column before the slice; prefill (DCP
1 only, where ids are offsets into the local logits) gathers the selected pools'
scores from the logits it ranked and does the same. The selected set is
unchanged. Fixed shapes, no host sync, safe under CUDA graph capture. Upstream
vLLM keeps all 512 pools (a 2,051-wide index); this keeps the engine's 511-pool
layout and only changes which pool is dropped.
Usage: port-kpool-col511-lowest-glm53.py VLLM_DIR
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


patch('model_executor/layers/sparse_attn_indexer_kpool.py', [
    (
"""
# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _use_b12x_kpool_indexer() -> bool:
""",
'''
# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _move_lowest_selected_pool_last(
    pool_topk: torch.Tensor, pool_scores: torch.Tensor
) -> None:
    """kpool-col511-lowest: the decode branch keeps pool_topk[:, : select_k - 1], and no
    b12x top-k route orders its output, so move each row's lowest-scoring selected pool
    into the last column (swap). The selected set is unchanged. Fixed shapes, no host
    sync (CUDA-graph safe). Unfilled slots (-1, score -inf) are lowest and go last first.
    """
    last = pool_topk.shape[1] - 1
    low = torch.argmin(pool_scores, dim=1, keepdim=True)
    low_ids = torch.gather(pool_topk, 1, low)
    pool_topk.scatter_(1, low, pool_topk[:, last:].clone())
    pool_topk[:, last:] = low_ids


def _prefill_selected_pool_scores(
    logits: torch.Tensor, pool_topk: torch.Tensor, row_starts: torch.Tensor
) -> torch.Tensor:
    """kpool-col511-lowest (prefill): top_k_per_row_prefill returns pool ids relative to
    each row's start and leaves rows with exactly select_k candidates in index order, so
    gather the selected pools' scores from the logits it ranked. Unfilled slots (-1) get
    -inf. Fixed shapes, no host sync.
    """
    ids = pool_topk.to(torch.int64)
    cols = ids.clamp(min=0) + row_starts[: ids.shape[0]].to(torch.int64)[:, None]
    scores = torch.gather(logits, 1, cols.clamp(max=logits.shape[1] - 1))
    return torch.where(ids >= 0, scores, torch.full_like(scores, float("-inf")))


def _use_b12x_kpool_indexer() -> bool:
''',
        1,
    ),
    (
"""                    row_starts=chunk.cu_seqlen_ks,
                )

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
""",
"""                    row_starts=chunk.cu_seqlen_ks,
                )

            if index_kpool > 1 and positions is not None and dcp_world_size == 1:
                # kpool-col511-lowest (prefill): the slice below keeps select_k - 1 pools.
                _move_lowest_selected_pool_last(
                    pool_topk,
                    _prefill_selected_pool_scores(logits, pool_topk, chunk.cu_seqlen_ks),
                )
            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
""",
        1,
    ),
    (
"""                cp_kv_cache_interleave_size = int(
                    cfg.parallel_config.cp_kv_cache_interleave_size
                )
            topk_scores = None
            if dcp_world_size > 1:
                topk_scores = torch.empty(
                    (num_rows, select_k),
                    dtype=torch.float32,
                    device=pool_topk.device,
                )
            _run_b12x_paged_topk(
                q_fp8=padded_q_quant_decode_tokens.reshape(
                    num_rows, *padded_q_quant_decode_tokens.shape[2:]
""",
"""                cp_kv_cache_interleave_size = int(
                    cfg.parallel_config.cp_kv_cache_interleave_size
                )
            # kpool-col511-lowest: request the selected scores on every DCP size.
            topk_scores = torch.empty(
                (num_rows, select_k),
                dtype=torch.float32,
                device=pool_topk.device,
            )
            _run_b12x_paged_topk(
                q_fp8=padded_q_quant_decode_tokens.reshape(
                    num_rows, *padded_q_quant_decode_tokens.shape[2:]
""",
        1,
    ),
    (
"""                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
""",
"""                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            )
            _move_lowest_selected_pool_last(topk_dst, topk_scores)
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
""",
        1,
    ),
], marker='kpool-col511-lowest')



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

print('GLM kpool indexer discards the lowest-scoring selected pool (kpool-col511-lowest)')
