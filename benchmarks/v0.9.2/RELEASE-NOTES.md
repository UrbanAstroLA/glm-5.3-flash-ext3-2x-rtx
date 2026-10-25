# v0.9.2 — request-owned caches and faithful expert computation

Five correctness fixes from **Michael M. (@UrbanAstroLA / AstroFella)**, covering kpool cache ownership, prefill chunk boundaries, sparse-attention selection, router precision, and routed-expert activation limits.

## Changes

- **Keep each request's kpool tail cache private and graph-safe.** Pass token positions into hybrid attention metadata, address the tail ring by position, mask null-block rows, and update the persistent mapping read by CUDA graph replay. This prevents tail writes from sharing the null block or using stale block IDs. [PR #7](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/7).
- **Complete pools correctly across unaligned prefill chunks.** Pool members from an earlier chunk are read from the same request's tail ring, rather than leaving a pool unwritten or using the previous request's tokens. This repair requires the tail-ownership fix above and is applied after it. [PR #11](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/11).
- **Discard the lowest-scoring selected pool.** The 511-pool layout now removes the least useful of the 512 selected pools, rather than whichever pool happens to occupy column 511. The reorder preserves the selected set before removal. It applies to B12x decode and DCP1 prefill. [PR #8](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/8).
- **Restore FP32 router output on SM12x.** Enable the requested FP32-output GEMM instead of rounding router logits to BF16 and then casting them to FP32. This ports upstream vLLM #54048. [PR #9](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/9).
- **Honor the routed-expert SwiGLU limit.** Pass the model's `swiglu_limit=10.0` to both EXL3/B12x planning paths. SwiGLU was already present; its model-specified clipping limit was missing from the routed experts. Dense MLPs and shared experts already honored it. [PR #10](https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/10).

Source-history inspection also finds the missing clamp argument in the adapter used by the original v0.1.0 recipe; this was not introduced by the recent cache fixes. Historical images were not rerun.

Serving defaults remain K4, DFlash2 K3, TP2 experts, DCP1 with MLA ownership, FP8 cache, and a 1M request limit. The launcher now uses the v0.9.2 image.

## Review and validation

The five PRs were reviewed together, including the #7 → #11 dependency, persistent CUDA graph buffers, score/ID alignment after DCP decode merge, prefill fallbacks, and B12x API support. Overlapping Dockerfile and provenance additions were reconciled while retaining all five ports.

The combined release image was built without GPU access; embedded compatibility and structured-output checks passed. All five supplied CPU regression gates passed (9 checks), with no GPU devices, no network, and no model loading. Release-specific CPU reports are stored alongside these notes. No local GPU serving or benchmark runs were performed because the release host's hardware was busy.

**Contributor GPU evidence** (reported in the linked PRs; 2× RTX PRO 6000 Blackwell / SM120, K3.25 weights):

- Tail ownership: zero null-block, foreign, or attention-block tail writes with the fix over 104 requests, versus 98% of writes going to the null block before it.
- Pool removal: no changes to the 512-pool selected sets and no non-lowest pool left in column 511 across 517,044 tiled decode rows, 145,112 fused decode rows, and 15.6 million prefill rows under the shipped graphs.
- Router precision: all 29.8 million checked logits matched the stated tolerance against an FP64 reference. The previous BF16 rounding changed the selected expert set on 3.15% of captured token/layer rows.
- SwiGLU: served kernel cache entries confirmed the limit reached the kernels. A deliberately smaller limit of 1 demonstrated reachability; how often the configured limit of 10 engages was not measured.
- Unaligned chunks: all 219 checked straddling pools matched an independent recompute byte-for-byte; 1.53 million aligned control pools had no mismatches.

**Limits:** these are correctness repairs, with no new end-to-end performance or quality-gain claims. Contributor GPU validation used K3.25; the K4 default was not requalified for this release. DCP2 on GPU, AMD, and SM121 hardware were not tested. SM121 router dispatch is covered only by the CPU emulation gate. External-token/KV-connector starts are outside the unaligned-chunk repair's validation. The README's performance tables remain historical v0.8.0 results.

## Upgrade

Pull the repository changes, stop your old recipe container, and run `./start.sh`.

Image: `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.9.2`.

Image index: `sha256:ea0430aa772e8cedfef1cdf80c30f4962ba43bb0991ac329950067ba1375f3e2`. Image source revision: `e7debee1fc3b3d0e75b3a2ede279e6be164c4c3e`.
