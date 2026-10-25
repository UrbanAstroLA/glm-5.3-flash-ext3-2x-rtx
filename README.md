# GLM-5.3 Flash EXL3 4bpw + DFlash2 on 2× RTX PRO 6000

A two-GPU daily driver: Brandon's higher-quality uniform-K4 checkpoint, a
**2M-token KV pool with 1M-token requests**, and faster reasoning-coding decode
than v0.7.1 on the same checkpoint.

This recipe serves [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
(formerly `GLM-5.3-Flash-EXL3-4bpw`) with the [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
drafter on two PCIe-connected SM120 GPUs.

`v0.9.2` fixes request ownership of kpool tail caches, pooling across unaligned
prefill chunks, and removal of the lowest-scoring selected pool. It also restores
FP32 router output on SM12x and passes the model's SwiGLU limit to routed experts.
See the [release notes](benchmarks/v0.9.2/RELEASE-NOTES.md) for contributor GPU
evidence, CPU checks, and validation limits. K4 was not requalified for this release.

`v0.9.1` keeps the newest one to three tokens in DCP1 sparse-MLA selections
below 2,044 tokens of context. This fixes short-context decode and early
sparse-prefill rows under the default MLA ownership profile. See the
[release notes](benchmarks/v0.9.1/RELEASE-NOTES.md).

`v0.9.0` fixes kpool tail-cache addressing and speculative ring corruption,
reports cached prompt tokens in API usage, and adds opt-in aligned DFlash2
prefix-cache lookup. See the [release notes](benchmarks/v0.9.0/RELEASE-NOTES.md).
The performance and quality tables below are the v0.8.0 baseline.

`v0.8.0` replaces decode-context parallelism with **per-layer MLA ownership**,
moves to the current [SparkInfer/B12x fork](https://github.com/tpurtell/sparkinfer-glmrt),
stores GLM's NoPE MLA cache in a compact 528-byte record, and removes most of
the per-request block waste in vLLM's hybrid KV pool.

## The numbers

All measurements below use the v0.8.0 release image on two RTX PRO 6000 Blackwell GPUs at a
**400 W limit each**. The v0.7.1 column is the published v0.7.1 image serving
the same K4 checkpoint with its defaults (DCP2, EP2, DFlash2 K5, 256K limit).

| Measurement | v0.7.1 on K4 | **v0.8.0** |
|---|---:|---:|
| Reasoning-coding score (mean of C1/C2/C4 medians) | 99.6 tok/s | **108.6 tok/s** |
| Reasoning coding, C1 per request | 131.4 | **140.9** |
| Reasoning coding, C4 per request | 67.0 | **77.0** |
| KV pool reported by vLLM | 335,088 @ 256K limit | **1,993,771 @ 1M limit** |
| Prefill, 128K prompt | 4,549 tok/s | **4,984 tok/s** |
| Seven-workload C1 blend | not measured | 149.8 tok/s |
| 1M-token six-needle retrieval | not supported | **6/6** |

The **reasoning-coding score** is v0.8.0's decision metric: four
realistic coding prompts with thinking enabled, temperature 1.0, 2,048 output
tokens, measured at one, two and four concurrent requests. Every default below
was chosen by it. Full tables, ranges and raw receipts are in
[`benchmarks/v0.8.0/RESULTS.md`](benchmarks/v0.8.0/RESULTS.md); the A/B arms
behind each decision are in [`benchmarks/v0.8.0-dev/`](benchmarks/v0.8.0-dev/README.md).

## Quick start

You need Linux/amd64, Docker with the NVIDIA Container Toolkit, two SM120 GPUs
with about 96 GiB each, a CUDA 13-capable driver, about 170 GiB of model
storage, and the Hugging Face `hf` CLI.

```bash
git clone https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx.git
cd glm-5.3-flash-ext3-2x-rtx
./download.sh
./start.sh
docker logs -f glm53-flash-exl3-b12x-vllm
```

The OpenAI-compatible endpoint is `http://127.0.0.1:8001/v1`. The container
reports healthy only after its startup warmup has exercised the release traffic
shapes. To build instead of pulling the published image:

```bash
IMAGE=glm53-flash:local ./build.sh
IMAGE=glm53-flash:local ./start.sh
```

**Upgrading from an older checkout.** Brandon renamed his repository. The Hub
redirects the old name, and the new revision `a5fee92` carries the same 120
weight shards as the old `4739eb1` snapshot. If you already have the old
snapshot, hard-link its blobs into the new cache folder before downloading so
only a few megabytes of metadata are fetched:

```bash
HUB=~/.cache/huggingface/hub
mkdir -p $HUB/models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/blobs
for b in $HUB/models--brandonmusic--GLM-5.3-Flash-EXL3-4bpw/blobs/*; do
  ln -n "$b" $HUB/models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/blobs/ 2>/dev/null
done
./download.sh
```

## Default configuration

| Knob | Default | Why |
|---|---:|---|
| `MODEL_PROFILE` | `k4` | Brandon's uniform-K4 checkpoint; `k325` and `k3` remain available |
| `MAX_MODEL_LEN` | `1048576` | 1M-token requests |
| `GLM53_MLA_OWNERS` | `split:25` | MLA layers 3–23 on GPU 0, 27–43 on GPU 1; replaces DCP2 |
| `DECODE_CONTEXT_PARALLEL_SIZE` | `1` | DCP is off |
| `ENABLE_EXPERT_PARALLEL` | `0` | TP2 routed experts: 108.3 vs 103.5 for EP2 |
| `DFLASH_TOKENS` | `3` | 108.3 vs 107.6 for K5, with a 5% larger pool |
| `KV_CACHE_PROFILE` | `fp8` | FP8 MLA cache in the 528-byte NoPE record (`NOPE_RECORD=1`) |
| `LANGUAGE_MODEL_ONLY` | `1` | Vision off; see options below |
| `GLM53_EMBED_SPLIT` | `0.17` | 17% of embedding rows on GPU 0, 83% on GPU 1 |
| `GPU_MEMORY_UTILIZATION` | `0.970` | Measured post-benchmark headroom is 2.1 GiB on the busier GPU |
| `MAX_NUM_SEQS` | `16` | Scheduler slots |
| `CHAT_TEMPLATE` | `zai` | Z.ai's corrected template; Brandon's checkpoint template drops images |

Sixteen scheduler slots do not mean sixteen 1M requests. vLLM reports
1,993,771 request-equivalent tokens (1.90 × 1M): its capacity counts each
maximum-length request's MLA pages plus the KDA recurrent-state checkpoints
every active request holds regardless of length. The C16 client sweep with
short prompts overlapped all 16 streams.

## Launch options

The v0.8.0 options below were validated for startup, sanity answers, teacher-forced NLL and 8K/240K
needle retrieval; only the default received the full benchmark battery.

```bash
LANGUAGE_MODEL_ONLY=0 ./start.sh        # vision on: 16-image contract, 1,731,627-token pool
KV_CACHE_PROFILE=nvfp4 ./start.sh       # NVFP4 MLA cache: 2,621,440-token pool
MAX_NUM_SEQS=12 ./start.sh              # 12 slots: 2,137,765-token pool
ENABLE_EXPERT_PARALLEL=1 ./start.sh     # EP2 experts: better at high concurrency, slower at C1
DFLASH_TOKENS=5 ./start.sh              # faster C1, slower C2–C4, smaller pool
SPECULATIVE_METHOD=mtp ./start.sh       # checkpoint MTP layer instead of DFlash2
GLM53_DFLASH_BOUNDARY_LOOKUP=1 ./start.sh  # opt-in: reuse the aligned DFlash2 prefix
```

| Option | Pool at 1M | NLL (lower is better) | Needles 8K/240K | Other checks |
|---|---:|---:|---:|---|
| Default (FP8, vision off) | 1,993,771 | 1.1786 | 6/6 | full battery |
| Vision on | 1,731,627 | 1.1779 | 6/6 | 1, 4 and 16 images pass; 17 rejected |
| NVFP4 cache | 2,621,440 | 1.1765 | 6/6 | — |

`GLM53_DFLASH_BOUNDARY_LOOKUP=1` (DFlash2 only, default off) stops every KV
group from dropping its last matching block on a prefix-cache hit. Without it,
the MTP-only group scoping does not recognize the DFlash drafter, vLLM's
all-group fallback marks every group as EAGLE, and each request recomputes
about two 3,584-token blocks of an already-cached prefix. On a downstream
TP2/DCP2 profile, repeated 31K and 62K prompts reused 28,672 and 57,344
tokens instead of 21,504 and 50,176; cached TTFT fell from 2.37 s to 0.93 s
and from 2.72 s to 1.35 s. Hits still land on the DCP2 grid (7,168 tokens).
Greedy outputs stayed correct, and draft acceptance did not drop on cached
requests. These measurements were reported by the PR author on the downstream
DCP2 profile; they are not benchmarks of this recipe's default MLA ownership.

API usage includes `prompt_tokens_details.cached_tokens` by default, so clients
can see how many input tokens were reused from the prefix cache.

`MODEL_PROFILE=k3` and `MODEL_PROFILE=k4` pin both the repository and its
revision. For another checkpoint, set `MODEL_ID` and `MODEL_REVISION` together;
the launcher rejects half-overrides so it cannot combine one model with
another model's commit.

The NLL column is `scripts/test-prompt-nll.py`: the mean negative
log-likelihood per token of fixed prose, code and a 17.5K-token back-reference
document, scored by the serving stack itself. It detects numerical damage in
the cache, attention or expert paths without sampling noise; the v0.7.1 image
scores 1.1743 on the same frozen text ([receipts](benchmarks/v0.8.0/nll-frozen/)). NVFP4 is indistinguishable from FP8 on this
probe and on the needles, but the probe tops out near 17.5K tokens and
Brandon's own teacher-KLD runs measured NVFP4 at roughly twice FP8's KLD
(0.055 vs 0.025), so FP8 remains the default.

Checkpoint MTP measured well below DFlash2 on the decision metric (91.4 static
K3, 72.3 adaptive, vs 104.2 for DFlash2 K3 on the same build) at equal
acceptance: MTP drafts sequentially through a full MLA+MoE layer per token.

## How the two GPUs are used

GLM-5.3 Flash has 45 layers: 34 KDA linear-attention layers and 11 MLA
sparse-attention layers, every one with 288 routed experts plus a shared
expert, wrapped in four-stream mHC residuals.

- **MLA layers are owned, not sharded.** Each MLA layer's projections, sparse
  indexer and latent cache live on one GPU and run all 64 heads there. The
  other GPU feeds zeros into the all-reduce that already follows attention, so
  the per-layer communication is unchanged and DCP's query gather, top-k
  exchange and LSE merge disappear.
- **KDA layers stay TP2.** Their recurrent state is per head, so head sharding
  splits weights and state evenly with no extra traffic.
- **Experts are TP2.** Every GPU holds half of every expert's intermediate width.
- **The embedding table is split unevenly.** GPU 0 owns six MLA layers and GPU 1
  five, so GPU 1 takes 83% of the token-embedding rows. Both GPUs then run out
  of KV budget at the same block count.

The per-GPU ledger the server logs at startup:

| GiB resident | GPU 0 | GPU 1 |
|---|---:|---:|
| MLA attention (owned layers) | 1.40 (3, 7, …, 23) | 1.16 (27, 31, …, 43) |
| KDA attention (TP2) | 4.40 | 4.40 |
| Token embedding (uneven rows) | 0.20 | 0.98 |
| LM head, shared experts, dense MLP, router (TP2) | 2.09 | 2.09 |
| DFlash2 draft (TP2) | 1.53 | 1.53 |
| Routed experts, EXL3 (TP2) | ≈71.6 | ≈71.6 |
| KV pool | 7.39 | ≈6.1 |

### Where the pool came from

| Change | Pool effect |
|---|---|
| DFlash2 draft window stored inside the MLA slot tensors (896/1088-token pages) instead of separate 128-token tensors | draft blocks per request 49 → 7 |
| 528-byte GLM_NEXT MLA record instead of the 656-byte DeepSeek record with an unused RoPE tail | +13.5% |
| DFlash2 K3 instead of K5 (two fewer KDA state checkpoints per request) | +5% |
| Skip vLLM's CUDA-graph dry capture, which reserved 0.71 GiB for a 0.23 GiB pool | +0.48 GiB per GPU |
| Utilization 0.95 → 0.97, vision off | +1.9 and +0.5 GiB per GPU |
| Uneven embedding split | +3% |

## Optimization review

A C1 reasoning-decode profile of v0.7.1 on this checkpoint split GPU time into
BF16 dense projections 39%, routed experts 35%, communication 9%, copies and
elementwise 7%, and attention plus indexer only about 4%. What that led to:

- **Topology.** Because attention is cheap, DCP2's value was capacity balance,
  not speed. MLA ownership removed DCP's exchanges; with the fixed capture path
  it measured +6% over DCP2 on the old fork.
- **Speculation.** Routed-expert time scales with the tokens each verification
  step touches, and reasoning acceptance is low (≈50% at K3). K3 beat K5 on
  C2/C4 and K7 lost everywhere; MTP lost to DFlash2.
- **Experts.** TP2 beat EP2 by 4.6% on the metric.
- **Dense GEMMs.** cuBLAS already runs the small-M BF16 projections at 85–88% of
  HBM bandwidth; cuBLASLt was no faster. No change.
- **Fork update.** The current B12x fork plus the fixes above is +9% over the
  published v0.7.1 image at a 4× longer request limit.

Two defects were found and fixed along the way, both now covered by gates:

- With DCP off, vLLM's breakable CUDA-graph capture left eager kernels in
  flight while the allocator recycled their memory, faulting intermittently at
  startup. DCP2's blocking collectives had masked it.
  `patches/port-glm53-breakable-capture-sync.py` synchronizes those segments.
- The first port to the current fork kept a head-major output view for
  sparse-MLA **prefill**, which the new extend kernel no longer honors. Short
  chats still looked fine, but teacher-forced NLL rose from 1.17 to 3.71 and
  long-context needles failed. Extend plans now request token-major output.

## Quality

| Check | Result |
|---|---|
| Teacher-forced NLL (vs 1.1743 for v0.7.1) | 1.1786 |
| API tool-choice constraints | 16/16 |
| Retrieval, thinking closed, at 8K / 240K / 1M | 3/3, 3/3, 3/3 |
| Six needles in one 1M-token prompt | 6/6 |
| Exact boundary: 1,048,288-token prompt + 256 output tokens | pass |
| Seven content contracts | 20/21 |
| Orchid exact-count repetition | 5/5 |
| Tool-eval-bench, 88 cases at C8 (v0.7.0 on K4: 157/176) | 159/176, Hard Mode 34/38 |
| Two concurrent 900K-token requests plus eight short chats | all complete, no OOM ([receipt](benchmarks/v0.8.0/stress-2x900k-c8.json)) |

## What differs from stock vLLM

Beyond the v0.7 composition (DFlash2 port, GLM EAGLE3 taps, EXL3 routed-expert
loading, ReplaySSM, B12x sparse MLA, PCIe one-shot all-reduce, mHC fusion and
the startup warmup gate), v0.8.0 adds:

- MLA layer ownership with owner-only weights and cache, an uneven embedding
  split, and a per-GPU placement ledger (`port-glm53-layer-owner.py`).
- DFlash2 draft cache co-owning MLA slot tensors (`port-glm53-draft-slots.py`).
- 528-byte GLM_NEXT MLA records (`port-b12x-glm-next-records.py`).
- The current fork's prepared plan/bind/run APIs (`port-b12x-latest-api.py`).
- Capture-time synchronization for breakable CUDA graphs and an optional skip
  of the graph-memory dry capture.

[PROVENANCE.md](PROVENANCE.md) records every immutable input and file hash.
The four EXL3 source files the build previously copied out of an 11 GB image
are vendored under `vendor/exl3-source/`.

The K3 and K3.25 profiles remain selectable but were not requalified for
v0.8.0; their receipts under `benchmarks/v0.7.0-k325/` describe v0.7.x.

## Thank you

Thanks to **Michael M. (@UrbanAstroLA / AstroFella)** for the kpool cache fixes,
DCP1 sparse-MLA tail investigation, router precision and routed-expert activation
fixes, reproductions, and validation.

Huge thanks to **Brandon** for the K4 quant, his teacher-logit dataset and KLD
receipts, and his public recipe. Thanks to **Inco AI / Z-Lab** for DFlash2,
**Z.ai** for GLM-5.3 Flash, **cstechdev** for the GLM day-zero image, the
**vLLM** and **B12x/SparkInfer** contributors, and the ExLlamaV3 authors whose
trellis work underpins EXL3. Thanks to **Samuel Cardillo** for the ReplaySSM
reproducer that remains a regression gate.

Recipe code is Apache-2.0. Model licenses apply separately: see Brandon's model
card for the K4 checkpoint's license, and note the DFlash2 checkpoint is
published under **CC BY-NC-ND 4.0 for research and evaluation**; contact Inco AI
for commercial licensing.
