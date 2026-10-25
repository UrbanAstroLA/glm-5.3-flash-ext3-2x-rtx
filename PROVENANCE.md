# Provenance

This recipe consumes finished Hugging Face target and draft checkpoints and composes pinned runtime artifacts. It contains no calibration corpus, quantization runner, writer checkpoint, or intermediate quant files.

## Immutable inputs

| Component | Immutable source |
|---|---|
| Served target (v0.8.0 default) | `brandonmusic/GLM-5.3-Flash-tr3-4bpw@a5fee929cf4888b1824323e33e8a19b60129e025` (renamed from `GLM-5.3-Flash-EXL3-4bpw`; same 120 weight shards as `4739eb1`) |
| Chat template (all profiles) | `zai-org/GLM-5.3-Flash-BF16@a5b45eb41df6402735dedc900be14a42e8d5e538` `chat_template.jinja`, SHA-256 `0c4099f3…c5`, vendored as `templates/glm53-zai-a5b45eb.jinja` |
| B12x fork (v0.8.0) | `tpurtell/sparkinfer-glmrt@7fcc094edcc93af61fdfbe14300100e3204363ea` |
| Supported K3.25 target | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1@0490d2f708b12145f6516555ab066aaeb401cd21` |
| Original v0.7.0 benchmark target | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1@701cd7456c13d87bf0147ad946f828a999afb59c` |
| Supported uniform-K3 target | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@1e4abd26e4e1e8d58d81fbd557d6c4099352fe63` |
| Supported uniform-K4 target | `brandonmusic/GLM-5.3-Flash-tr3-4bpw@aba59d2175e1ee2887ae0ae1300ba848b1deed84` |
| Target source | `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e` |
| Corrected chat-template source | `zai-org/GLM-5.3-Flash-BF16@a5b45eb41df6402735dedc900be14a42e8d5e538` |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| GPTQModel quant writer | `tpurtell/GPTQModel@0565af7ce20a93df9bbc0e5563d7c6f60916f41a` |
| GPTQModel compact-config follow-up | `tpurtell/GPTQModel@a64900815b30ef01c2221b2788701a7986e50491` |
| K3.25 GPTQModel writer | `tpurtell/GPTQModel@a053382584fa58cba7bf212ef1b829d08b29b2c0` |
| K3.25 readable-plan follow-up | `tpurtell/GPTQModel@0b6734a8cfeabecae78a7a82f7fc82ec97bddfc5` |
| GLM/vLLM base | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| vLLM in base | `0.1.dev20051+g487ecf187` |
| vLLM DFlash2 delta | `vllm-project/vllm@b389ac29465b33f9e9c534df221ea3c129e9793f` (PR `#52816`) |
| EXL3 runtime source files | vendored `vendor/exl3-source/` (byte-identical to `/opt/vllm` in `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32`) |
| EXL3 vLLM fork commit | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| B12x fork (v0.7.x) | `tpurtell/sparkinfer-glmrt@fe054789069579e19ae5ec21f880b397bcf6575b` |
| ReplaySSM base | vLLM PRs `#48792`, `#49847`, and `#49887`, ported onto the pinned vLLM commit |
| ReplaySSM mixed-graph repair | This repository's `c51c3856f7f8ba50af3b3a60ff48e7d6a1fa303c` |
| Dynamic-MTP graph fix | vLLM PR `#49652`, ported onto the pinned vLLM commit |
| Runtime stack | Torch 2.13, CUDA 13, CUTLASS DSL 4.6.2 |

## Published v0.9.1 release artifacts

The v0.9.1 image is built from PR #6 merge commit
`84572a81e0092796d396acf4c3f22d6aabc3724b`. No image input changed after
that commit; the release follow-up updates the launcher, regression gate,
and documentation.

| Artifact | Immutable identity |
|---|---|
| Image index | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07` |
| Linux/amd64 manifest | `sha256:0be9045a2ae51c624d567ec916763eef62f7f8207fd1f5fc92253d328e123668` |
| Image source revision | `84572a81e0092796d396acf4c3f22d6aabc3724b` |

The actual kpool expansion and DCP1 selection/masking pipeline reproduced
the newest-token loss in v0.9.0. With the patch, all 54 row checks passed
across 18 context lengths in contiguous, strided, and CUDA graph modes.
Release evidence and contributor measurements are recorded separately in
[benchmarks/v0.9.1/RELEASE-NOTES.md](benchmarks/v0.9.1/RELEASE-NOTES.md).

v0.9.1 file hashes:

| File | SHA-256 |
|---|---|
| `Dockerfile` | `563d95c7affaeffbd491675bd04718c97daef472bfd2c49b7c11efadc07d3799` |
| `start.sh` | `15b3629743140aa87a9285994b456ec5febc5359608523f366490c890978c283` |
| `patches/port-dcp1-kpool-tail-glm53.py` | `388f2b664fca7add857f314bfb0af8b5edaa72a6b3210e8fd1c15ee0f3917498` |
| `scripts/test-dcp1-kpool-tail.py` | `eaa559e83f74b83e88c4cac0426a88bb0ba6bae2a64ad7e3758dfbb587f11271` |

## Published v0.9.0 release artifacts

The v0.9.0 image is built from the combined PR merge commit
`c512b93ee4577f16e0e9396ad001d5203324bf15`. No image input changed after
that commit; the release follow-up updates the launcher, tests, and documentation.

| Artifact | Immutable identity |
|---|---|
| Image index | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:f36dfb876da9393c95ca738ed912ca73905d0211004ce5c09b9ae312ed0f5cc9` |
| Linux/amd64 manifest | `sha256:e57ab0cdb8548130f465c0d2ee530a66ffdd1c1fed1a42d351aaed3c1a88ec7a` |
| Image source revision | `c512b93ee4577f16e0e9396ad001d5203324bf15` |

The full image build and its compatibility/structured-output probes passed.
The upstream kpool GPU regression suite passed 33 tests and skipped one
ROCm-only test. The pinned test source and release-specific receipts are
recorded in [benchmarks/v0.9.0/RELEASE-NOTES.md](benchmarks/v0.9.0/RELEASE-NOTES.md).
The v0.8.0 performance tables remain historical measurements.

v0.9.0 file hashes:

| File | SHA-256 |
|---|---|
| `Dockerfile` | `251a48698d4a565d883c93dd5f063e29fe3e5d0d21db4579cbcc76c9a434395d` |
| `start.sh` | `e1196cbcd6b4744cb41fdebf3d12d8586dbea464e702af5de47617837efe0e5e` |
| `patches/port-dflash2-boundary-prefix-cache.py` | `c6d7a149199bb73ad641b3f26f50f536a96d74db93fdeed1fbaacdfc3865d992` |
| `patches/port-kpool-seed-stride-glm53.py` | `4959e5762f97c70ed074397e09c6835fdfcb4768c2002ea5d0670ea42ddec624` |
| `patches/port-kpool-spec-ring-glm53.py` | `5a72fae2a48e67bce8953c76c0b2b78aafca8e663389d695910790cc6eac53f5` |

## Published v0.8.0 release artifacts

At publication, `v0.8.0` and `latest` resolved to the same OCI index, built from recipe commit
`48927a5c4e358c9b74faa41dc5627db635f5d45c`; no image input (Dockerfile,
patches, container scripts, template, vendored sources) changed after it.

| Artifact | Immutable identity |
|---|---|
| Image index | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:e4d37a01da91ec590df78af13efa9d256fc306a5a88df09908ea12ebefa32423` |
| Linux/amd64 manifest | `sha256:3a6fc6e71b6286b2494b70ac716de30d58e71f33a779eb735895f3570eedba29` |
| Image source revision | `48927a5c4e358c9b74faa41dc5627db635f5d45c` |

The full battery in `benchmarks/v0.8.0/` ran on this exact image.

## Upstream kpool tail fixes (after v0.8.0)

Ports of two GLM-5.3-Flash fixes merged in vLLM after this recipe's base was cut. Both images up to and including
v0.8.0 fail the four upstream regression tests for them in `tests/kernels/test_kpool_decode_update_batched.py`
(29 pass); with both ports applied, all 33 pass (one ROCm-only test skips).

- `port-kpool-seed-stride-glm53.py` ports vllm-project/vllm#57477. The tail cache aliases the indexer cache with the
  indexer's padded block stride (`tail.stride(0)` is 71808 elements on the v0.8.0 layout, not the dense 2048), but the
  prefill seed kernel addressed it densely: every prefill left its own tail block unseeded and wrote 2 KB of raw K and
  gate rows into another block's indexer region. The kernel now addresses blocks through `tail.stride(0)` /
  `tail.stride(1)`, as the decode kernel already did, and the wrapper asserts the view layout.
- `port-kpool-spec-ring-glm53.py` ports vllm-project/vllm#58454 and must run after the seed-stride port (it refuses
  otherwise, writing nothing). The tail ring held exactly `index_kpool` (4) slots per request, but drafts are stashed
  before acceptance: when a pool-completing draft is rejected, the drafts behind it have already overwritten that pool's
  committed keys and the redo compresses the pool from corrupted slots. This affects any speculative method with 2+
  draft tokens once context exceeds `index_topk`. The ring now holds `kpool * next_power_of_2(cdiv(kpool + num_spec,
  kpool))` slots (8 for 3 drafts, 16 for 5-7, unchanged at 4 without speculation). Slot mapping, cache shape, allocator
  and page size all derive from the spec's `block_size`; the larger ring fits inside the existing page padding, so KV
  capacity was unchanged in the contributor's K3.25 run (4,707,515 tokens before and after).
- `port-dcp1-kpool-tail-glm53.py` keeps the kpool tail in DCP1 sparse-MLA selections. The indexer writes the selected
  pools to columns `0..2043` and the request's incomplete trailing pool (the newest 1-3 tokens) to the fixed columns
  `2044..2046`; the DCP1 branch of `B12xMLASparseImpl.forward_mqa` then masks every column at or past
  `min(causal length, 2048)`. Below ~2,044 tokens of context, whenever the length is not a multiple of 4, the newest
  1-3 tokens were dropped from every MLA layer (the same applies to sparse-path prefill rows in a batch whose longest
  prefill exceeds 2,048 tokens). DCP2 compacts its selection and is not affected. The patch adds one Triton kernel, run
  just before the existing mask, that stable-compacts a row's valid entries to the front only when the mask would drop
  one; every other row is untouched (byte-identical selections). It refuses unless the target file matches the expected
  hash. Measured on K3.25, 6 GPQA prompts x 2,600 decoded tokens, speculation off: decode-vs-prefill KL on the same
  token ids fell from 0.066 to 0.010 for positions below 2,044, and from 0.031 to 0.019 beyond 2,048, where tokens
  decoded under the bug still sit in the cache.

## Correctness fixes after v0.9.1

Each script edits the installed vLLM tree at the end of the second patch chain, validates every anchor before
writing, compiles the result, replaces files atomically and is a no-op when already applied. Each has a CPU
regression gate in `scripts/`; `--expect-unpatched` reproduces the defect on v0.9.1.

- `port-kpool-tail-positions-glm53.py` (upstream vllm-project/vllm#53906 and #59528): `MambaHybridModelState`
  now passes `positions` to `build_attn_metadata`, so the kpool tail group gets its circular mapping
  (`own_block * ring + pos % ring`) instead of the generic one, which sent tail writes from position 8 to the
  null block or to a block id left by the start-up warm-up. Rows on the null block get PAD, and the mapping
  is copied into the persistent per-group slot-mapping buffer so FULL CUDA graph replays read it. With
  positions present, the B12X sparse-MLA metadata builder takes causal lengths from `positions + 1` for
  batches that are not pure single-token decode, as upstream does. Gate: `scripts/test-kpool-tail-positions.py`.
- `port-kpool-unaligned-chunk-glm53.py` (requires `port-kpool-tail-positions-glm53.py` and refuses, writing
  nothing, without it): the prefill index-K writer pools batch rows t-3..t and assumed pool-aligned chunk
  starts, which the scheduler does not guarantee (budget minus draft slots, decode rows and other requests in
  the step). A pool straddling an unaligned start is now completed from the request's own tail ring, which
  is position-addressed only with the tail-positions port. Rows without a tail slot keep the old rule.
  Gate: `scripts/test-kpool-unaligned-chunk.py`.

- `port-kpool-col511-lowest-glm53.py`: the indexer keeps 511 of the 512 pools its top-k selects and no top-k
  route orders its output, so the discarded pool was whichever pool sat in column 511. Decode (any DCP size)
  and prefill (DCP 1) now swap each row's lowest-scoring selected pool into the last column before the slice.
  The selected set is unchanged; upstream keeps all 512 pools instead. Gate: `scripts/test-kpool-col511-lowest.py`.

- `port-router-fp32-out-sm12x.py` (upstream vllm-project/vllm#54048): GateLinear's fp32-output router GEMM is
  enabled on compute capability 12.x, where the router previously ran in bf16 and cast the result.
  Gate: `scripts/test-router-fp32-out-sm12x.py`.

## v0.8.0 runtime changes

v0.8.0 changes the served default to Brandon's uniform-K4 checkpoint and
replaces the attention topology. Every change below is a source-locked Python
port under `patches/` that fails on anchor drift and is applied idempotently in
the Dockerfile after the v0.7 ports.

- `port-b12x-latest-api.py` moves every B12x call site to the fork's prepared
  `plan`/`bind`/`run` API at `7fcc094e` (sparse MLA, DSA indexer, EXL3 fused
  MoE, PCIe all-reduce and DCP helpers, mHC, BF16 GEMV). Sparse-MLA **extend**
  plans request token-major output: the current fork's extend kernel no longer
  honors a head-major output view, and requesting one corrupted prefill
  attention (teacher-forced NLL 3.71 versus 1.17). Decode keeps head-major.
  The image restricts `VLLM_PLUGINS` to vLLM's LoRA resolvers because the
  fork's own vLLM plugins target a newer vLLM.
- `port-glm53-layer-owner.py` adds `VLLM_GLM53_MLA_OWNERS`. Each of the 11 MLA
  layers (projections, sparse indexer, latent and indexer cache) exists only on
  its owner GPU and runs all 64 heads unsharded; the other GPU contributes
  zeros to the existing post-attention all-reduce. KDA layers stay TP2 because
  their recurrent state shards cleanly by head. `VLLM_GLM53_EMBED_SPLIT` places
  an uneven share of the token-embedding rows on each GPU to balance the two
  KV pools. The patch also logs a per-GPU placement ledger and a per-group KV
  capacity breakdown.
- `port-glm53-draft-slots.py` stores the DFlash2 sliding-window cache inside
  the MLA slot tensors with a block size that divides the MLA block, instead of
  separate 128-token tensors. Draft blocks per request drop from 49 to 7–8.
- `port-glm53-graph-memory.py` lets `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
  skip the dry graph capture, whose 0.71 GiB estimate covered a 0.23 GiB pool.
- `port-glm53-breakable-capture-sync.py` synchronizes capture-time eager
  segments of vLLM's breakable CUDA graphs. With DCP off, in-flight eager
  kernels raced the allocator and faulted intermittently (MMU fault in a BF16
  add, located from a GPU core dump); DCP2's blocking collectives had hidden it.
- `port-b12x-glm-next-records.py` adds `VLLM_GLM53_NOPE_RECORD=1`: GLM-5.3's
  NoPE MLA latents are stored as the 528-byte GLM_NEXT record (512 E4M3 bytes
  plus four FP32 scales) instead of the 656-byte DeepSeek record with an unused
  RoPE tail. Decode and prefill outputs match the 656-byte path within BF16
  rounding (`scripts/test-glm-next-records.py`).

v0.8.0 file hashes:

| File | SHA-256 |
|---|---|
| `Dockerfile` | `cd6dc0af7031218a3b21244eecd07c3badda6e95f9d7c30d30efa1c27ed893fd` |
| `start.sh` | `b67eca0733c0c0e0bb6de3f695721b491f7e9249f2384c0b159b212efbe41773` |
| `model-profiles.sh` | `a4ccf85dc1e1dd75acdb8549ecaec82137c9f77b47e5bb6464d82f04ef5537f1` |
| `patches/port-b12x-latest-api.py` | `f9a5585872a834e91a3d5017a9f4cd8346065af40e8287e733ae993b57a56c59` |
| `patches/port-glm53-layer-owner.py` | `b3e8bf8890784890acdf84a76ae618adf99e3189e03251c155a25753a4d80d62` |
| `patches/port-glm53-draft-slots.py` | `d188439dc7102ff562a0d2f286ab6814e6a72c24bd2885d6e3f9f808b1bfe713` |
| `patches/port-glm53-graph-memory.py` | `10670cbe9041b4e8235511a44dbe65ef1c66a0bb689afc324406a551a2bff6cf` |
| `patches/port-glm53-breakable-capture-sync.py` | `b26d41a182f34b71b11c441f5e5db3f13764e72d87a3dbf435b0935d651140a6` |
| `patches/port-b12x-glm-next-records.py` | `e72dcbd8935cbc2f65118938c1e7baf929796e2ac28fffe33ecf9b385420ec8c` |
| `vendor/exl3-source/…/quantization/exl3.py` | `209769899a069615e7c8ace17d52515f89ffaf2c73a77532ee45f6de1919710c` |
| `vendor/exl3-source/…/mla/b12x_mla_sparse.py` | `9eb6daa3031c5cf6a03209ce54f0e5c2874a858caa04d89c75872db42ca228e9` |
| `vendor/exl3-source/…/layers/sparse_attn_indexer.py` | `4e91b4fc63c4d8472006a475a5c03849c80e2c4fdbe2f27603837c0302fe1c29` |
| `vendor/exl3-source/…/layers/mla_cache_format.py` | `f3b61e5c366837bca6b1c7039e8570e6884ab6bdb707eafd214e5a5fb28bcc2f` |

## v0.7.1 structured-output fixes

Two source-locked Python backports address distinct speculative grammar edges:

- vLLM [`c6e19b3be243` / #53046](https://github.com/vllm-project/vllm/commit/c6e19b3be243):
  validate speculative tokens after the reasoning-end marker before advancing
  the grammar. Real constrained-token failures outside that window still raise.
- vLLM [#52805](https://github.com/vllm-project/vllm/pull/52805), carried from
  our MIA and single-Spark GLM recipes: stop accept/validation batches at
  termination, make subsequent accepts no-ops, and clear termination on reset.

The source anchors match the installed v0.7.0 files exactly. Patchers reject
unknown input and validate already-patched content. The image build runs 14
CPU regressions against the installed methods; the original v0.7.0 image fails
five checks (three termination, two reasoning-window checks). No quantization,
attention, MTP/DFlash numerical kernels, dependency versions, model pins, or
serving parameters change. The launcher now selects the versioned image to
avoid silently reusing an older cached `latest`.

Focused live qualification uses `scripts/verify-structured-output-live.py`
(the upstream JSON trigger, C16 thinking on/off, normal strict JSON, and
ignore-EOS JSON with an explicit stop)
and the 145-case `scripts/verify-issue136-xgrammar-live.py` matrix at C8.
The latter retains its original, weaker ignore-EOS transport-only lane; the
former separately requires complete valid JSON and a `stop` finish state for
29 cases. Three verbatim upstream 500-token requests are additional diagnostics:
GLM can exhaust that budget on thinking, so truncated responses are recorded
explicitly rather than counted as complete-JSON passes. Five normal strict-JSON
requests and five ignore-EOS requests with an explicit closing-brace stop check
complete bodies. `ignore_eos=true` without a stop can continue after the grammar
terminates; the initial probe documenting that limitation is retained.
Performance/needle/vision/full tool-eval results are retained as historical
v0.7.0 measurements, not represented as newly rerun results.

The exact v0.7.1 image passed 14/14 installed-method CPU checks, 145/145 live
canary cases at C8, and 29/29 complete-JSON checks including C16 thinking
on/off. The final three verbatim upstream requests also completed. The server
remained healthy with zero FSM errors, tracebacks, ERROR lines, HTTP errors,
or post-ready JIT over all 201 requests including the retained initial probe.
See [the focused report](benchmarks/v0.7.1-xgrammar/README.md) for the initial
500-token truncation and ignore-EOS continuation caveats.

## Published v0.7.1 release artifacts

Both `v0.7.1` and `latest` were verified through anonymous GHCR reads against
the exact image used for live qualification:

| Artifact | Immutable identity |
|---|---|
| Image index | `sha256:fc6615ac0386dd03f3ef6380fba01489f48f7619b064157d5b95431989fd9ddc` |
| Linux/amd64 manifest | `sha256:0463bfa7385920b37575eb78e1286fe2d2b9d40ab8ab2e616be3a970faf957b2` |
| Image config | `sha256:f7b176ff4784d499c14af261627727d5433d6a541c1499e2f8761239c733da7a` |
| Image source revision | `9c35641652670bd216a4ad29495c3edd41f0828c` |

[Registry verification](benchmarks/v0.7.1-xgrammar/registry-verification.json)
binds both tags to those identities. The release tag additionally contains the
post-build live-test receipts, test-harness refinements, and documentation;
the runtime patches and Dockerfile remain identical to the image source.

## Historical v0.7.0 release artifacts

At v0.7.0 publication, the `v0.7.0` and `latest` tags resolved to the same immutable
OCI index:

| Artifact | Immutable identity |
|---|---|
| Recipe image | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.7.0@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629` |
| Linux/amd64 manifest | `sha256:5b0486d3ada90ee3c0d822baed55e1a4d65e06e6ecc78c66a40bd22bcfa9a891` |
| Image config | `sha256:507791943bfd239a34fea7ef276353b14a14c8d597a97aac982e7aed372be0d9` |
| Build/source revision | `92eec28c9ad4d681af5f4861b74811695bfbcfa1` |

The registry independently returned that index, manifest, and image config.
Its OCI labels pin the source revision above, B12x `fe05478…`,
DFlash2/vLLM `b389ac2…`, and version `v0.7.0`. The Git release tag includes
this post-build provenance; the serving code in that tag is unchanged from
the embedded image source revision.

The DFlash2 checkpoint is a 1B-parameter BF16 draft model and is not a standalone language model. Inco AI publishes it under CC BY-NC-ND 4.0 for research and evaluation; commercial use requires separate licensing. The target model and base-image licenses also apply independently of this Apache-2.0 recipe.

## Target identity and metadata repair

K3.25 revision `0490d2f…` is public and ungated. It contains 18 EXL3 shards,
the external 32.8 MB tensor manifest, the official multimodal processor,
tokenizer/generation metadata, and Z.ai's corrected chat template. Its compact
`config.json` does not duplicate the tensor manifest. The normal Hugging Face
cache was installed from the materialized local files, then verified at the
public revision without redownloading the quant.

Current target metadata hashes:

| File | SHA-256 |
|---|---|
| `config.json` | `6b477cfc1fbf8cdf3795c6389bc9712503e3f7c3889145036488ffab2b1a7781` |
| `chat_template.jinja` | `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` |
| `processor_config.json` | `aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed` |
| `tokenizer_config.json` | `926e1d0692d9f46940311494bd6de97f208e195c9150883c163f16b30c868ff4` |
| `generation_config.json` | `a07de3408f578c6a7ca8a1646aa91a41df55d539349fda15fb8b611eb007e9b7` |

The compact `config.json` deliberately does not embed the duplicate 32.8 MB EXL3 `tensor_storage` map. The complete tensor map remains in the external quantization manifests. The launcher validates the index, every referenced shard, the EXL3 manifest, official processor metadata, and DFlash2 architecture before starting Docker.

The compact K3.25 config is 13,180 bytes, parses as ordinary JSON and through
the release vLLM configuration path as `Glm5NextConfig` /
`Glm5NextForConditionalGeneration`.

Correction, 2026-09-06: the original K3.25 revision `701cd74…` actually shipped
template SHA-256 `34d5ee66…`, not the corrected upstream template previously
claimed here. Revision `0490d2f…` replaces only `chat_template.jinja` relative
to the then-current Hub head `59484d5…`, preserving every other remote file.
It is byte-identical to Z.ai `a5b45eb…`, including null-content handling,
tool-name coercion, and early exits in tool-result reordering checks.
The local cache uses a new immutable snapshot and reuses all 18 existing
weight blobs; old snapshots remain unchanged for benchmark reproduction.
All 35 files in the new snapshot passed `hf cache verify`, with missing and
extra files treated as errors. Offline `main` resolves to the new snapshot.

The corrected template passed 11 CPU-only identity/render assertions through
the release container's `AutoProcessor` API (which resolves to
`TokenizersBackend` here): default thinking/effort, null content, tool ordering
and invalid-ID fallbacks, reasoning continuity, and 16 image placeholders.
This is a template check, not a new vision-inference or tool-eval run.
[The sync receipt](benchmarks/chat-template-sync-k325-20260906.json) records
the exact identities and checks. The published benchmark receipts remain
historical results for their original target revisions.

The GPTQModel commits are listed only to establish how the public targets were
written and packaged. The earlier K3 work ran at `0565af7…`; the K3.25 one-shot
run used `a053382…`, with the compact-config work from `a649008…` already in
its ancestry. `0b6734a…` is a post-run readability improvement for published
mixed-tier plans, not a silent weight rewrite. The saver materializes EXL3
shells while streaming native tensors under their original keys, avoiding
duplicate BF16 materialization. GPTQModel does not ship in the serving image.

## Runtime composition

The Docker build copies only the qualified EXL3 loader/adapter files from the source image, fetches the pinned B12x fork, applies the existing ReplaySSM/dynamic-MTP series, ports the immutable upstream DFlash2 commit, then applies the narrow GLM/DFlash/DCP corrections under `patches/`.

The DFlash2 release work adds:

- GLM-5.3 target support for EAGLE3 hidden-state taps after mHC and the decoder-layer indirection expected by the upstream DFlash implementation.
- An independent hybrid KV group for the dense DFlash draft rather than forcing it into the target model's MLA/recurrent cache groups.
- Per-attention DCP scope: target attention remains DCP2, while the draft model, draft metadata, draft forward state, and draft CUDA graphs are replicated DCP1 on both ranks.
- A 128-token allocation block for the replicated DFlash sliding cache instead of the inherited 16-token page, reducing long-request shared-pool block-ID overhead.
- Draft-scratch exclusion from target prefix hashes and compatibility logic for the replicated draft sliding group.

The v0.6 B12x/EP2 work adds:

- EXL3 global-to-local weight loading for GLM's 288 global experts and 144
  local experts per EP2 rank. Non-local checkpoint weights are skipped before
  materialization; global top-8 route IDs and weights remain ordered.
- B12x `ep_moe` plan/bind/run integration with fixed decode and prefill
  workspaces, replicated input, and vLLM-owned final TP reduction.
- The public B12x full-rotation MCG Trellis EP fix at `611ffe8`, including
  graph-stable FP16 rotation, route, output, and barrier arenas.
- Migration from the retired `nsa_indexer` API name to the current B12x
  `dsa_indexer` contract and current RTX PRO 6000 Blackwell policy data.
- GLM mHC shape admission during engine warmup plus a container entrypoint
  that gates health on raw-greedy C1, rendered-chat C1, long-prefill, and four
  sampled C16 passes. This moves every observed route, DFlash, sampler,
  sparse-indexer, and mHC first-use compilation before the ready marker.

The K3.25 runtime update adds a second, projection-native EXL3 path without
replacing the qualified uniform-K3 path:

- It derives each expert's integral gate/up/down K value from the external
  EXL3 tensor manifest. The checkpoint-level `3.25` value is descriptive;
  executable projections remain exactly K3 or K4.
- It prepares B12x `ProjectionTrellisTierWeights` using the live-shape
  `trellis_t256_proj` selector. Uniform K3 and K4 checkpoints continue through
  the fixed `b12x_trellis` layout and keep their existing kernel policy.
- Under EP2, vLLM composes the immutable 288-global-to-144-local expert map
  with B12x's local-expert-to-tier map once while loading each layer. Decode
  and prefill then bind one precomputed global-route-to-tier map with no live
  allocation or route reordering.
- B12x `fe054789...` admits that larger route namespace, preserves the
  projection-mixed live tile selector, reduces sparse-index and scratch width,
  and fails closed on route-map dtype, extent, device, and contiguity drift.
  Its final CPU suite passed 74 tests, and the composed two-GPU release gate
  passed 47/47 mixed-Trellis, EP, attention, high-page, and PCIe tests.

Before GPU qualification, the composed image passed its build probe, the
projection port's complete idempotence pass, and a full-manifest regression
gate over the published K3 model: all 37,152 routed projections across layers
3--45 remained uniform K3 and selected the fixed-tier metadata path. These are
source/metadata gates, not substitutes for the required K3.25 GPU performance
and numerical qualification.

The pre-existing local work remains available:

- GLM K3 routed-expert/MTP mappings for the mixed EXL3 checkpoint.
- Vector-gated KDA ReplaySSM, compact state accounting, and request-lifetime adaptive MTP as an alternate speculative method.
- The bounded EXL3 prefill arena and GLM-specific B12x sparse-MLA, K-pool, DCP2, PCIe collective, query-projection, mHC, NVFP4, and FP8 cache ports.

### ReplaySSM mixed-batch repair and release gate

[Samuel Cardillo's corruption investigation](https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell/commit/1755c0f0c01b98463a7b87ab613a6c894b569298)
is a valuable downstream reproducer, but it used this recipe at `3bff1d5...`
(`v0.3.0`). That source predates `c51c385...`, which prevents compact
ReplaySSM from selecting vLLM's generic mixed prefill/decode CUDA graph. The
generic graph supplies synthetic rows whose shape can disagree with the live
GLM KDA prefill state; Samuel's fatal
`ReplaySSM prefill source/state row count mismatch` occurred under the same
rolling C4 mixed workload. Uniform decode graphs remain enabled.

The current port also includes the request-lifecycle corrections from vLLM
[#49847](https://github.com/vllm-project/vllm/pull/49847): draft-less rows stay
on ReplaySSM, a temporarily unscheduled request retains its pending GPU
acceptance and decode anchor, and preempted/recycled pages reset. vLLM
[#54103](https://github.com/vllm-project/vllm/pull/54103) documents a nearby
KDA concurrency hazard in which a strided `state_indices[:, 0]` was consumed
as contiguous. This CUDA port already passes that stride into its verify and
cursor kernels; the KDA prefix materializer now carries explicit strides for
all four row tensors as well and reports every row count on invariant failure.

The K3.25 release gate first compares the KDA prefix materializer with a Torch
reference while all four request-row inputs are deliberately non-contiguous.
It then runs `scripts/test-replayssm-stress.py` with ReplaySSM on:
exact 32,768-token prompts, C4, 40 thinking-off shared-prefix requests, 40
maximum-thinking shared-prefix requests, and 40 maximum-thinking unique-prefix
requests. It retains raw SSE, rejects repeated-subword streams, requires exact
needle retrieval with thinking off, checks server health after every phase,
and compares the same workload with full-state rollback. ReplaySSM and the
matched control each passed 120/120 requests with zero loops, errors, engine
deaths, or post-ready JIT; the strided CUDA materializer matched exactly with
maximum absolute error 0.0. ReplaySSM is therefore a qualified option. It is
not the DFlash default because the release workload measured higher C1 decode
with baseline rollback.

The newer FlashInfer work in vLLM
[#52928](https://github.com/vllm-project/vllm/pull/52928) is Mamba2-only and
currently requires `mamba_cache_mode=none`; the prefix-cache follow-up
[#54609](https://github.com/vllm-project/vllm/pull/54609) is still WIP. Neither
is a drop-in replacement for GLM-5.3's vector-gated KDA plus aligned prefix
cache, so this release keeps the narrow KDA port rather than importing an
unrelated backend.

Build-time probes check target/draft architecture recognition, the DFlash2 V2 speculator, GLM EAGLE3 support, EXL3 registration, uniform and per-projection Trellis plans, the EP route namespace, ReplaySSM/adaptive policy imports, B12x APIs, compact cache layouts, head geometry, and exact runtime versions.

Current v0.7 launcher/build hashes:

| File | SHA-256 |
|---|---|
| `Dockerfile` | `682fe68886e019edd9d6eff3ae7c239833e3f9fd067facd342ad50492cb7516d` |
| `build.sh` | `785958e0d667dfbd0141d59b4978d3909ed3c191f88b1daa7575c99badbd2452` |
| `download.sh` | `f952a707a412605ff99c3336d5e7040ee95a20797aca224efd045b273fa6e9a8` |
| `start.sh` | `fbd116da87ee01525f76a6016a30c4d9ef0f5aa8fb6188cbb5e6ea571aa88212` |
| `container/glm53-entrypoint.sh` | `c1de8b073277b8edfb5c85c7c8d83ee511593d77d271c26a7ddf4d1e1c5abc8d` |
| `container/glm53-release-warmup.py` | `dd4e3bc041f267d288748552248a412773a2e308691145a70616977a39558146` |

Changed/new serving patches in v0.7:

| File | SHA-256 |
|---|---|
| `port-exl3-glm53.py` | `4c306ddde7e44e5888b4b75d26046b22a3d6cf920dbc30001c8fb0574bfee3cc` |
| `port-exl3-projection-mixed-glm53.py` | `8be477cca588993040c879ca4f572f4c91af2b43fd58e68dbb355297e4391c00` |
| `port-exl3-prepared-dtype.py` | `0d77bdd107a385ac7df550d0141e18b9c86d9dbe4b0e8ed4906cd8d905f22973` |
| `port-b12x-glm-index-width.py` | `0ccce3229e38654f506fe3049293853812fe436c2012b86feabb16fe5427e9d8` |
| `port-glm-sparse-memory.py` | `df1cde80246045ae9ef8cb2e151a8679f52ac27f012347bc00a61b1c42ce4851` |
| `port-glm-prefill-jit.py` | `cd535cdc25f02fcf681c799c09a3275aeb7151c8fb7802b838b2e2a4cb550c98` |
| `port-glm-replayssm-conv-window.py` | `78620f40d5c99d75bcf1e43da8e4d5456e147fa91dbae4394a45c25ce004b944` |

The build executes every port idempotently against the fixed base image and
probes the resulting source/API contracts before committing the image layer.

## Qualification evidence

The published target revision was exercised through the normal launcher rather than a local override. Release gates include:

- A final K5 code-agent C1–C16 curve, existing-depth decode through 128K, cold
  prefill through 128K, and matched final-image uniform-K3 control.
- The seven-type GLMRT content blend with 21/21 semantic contracts passing.
- Focused B12x projection-mixed parity, EP map, rank-partial parity,
  allocation-free graph replay, sparse attention, high-page, and PCIe tests:
  47/47 pass on the target GPUs.
- Direct DFlash K5 mask-anchor, rejection rollback, sampling-position,
  request-map, and inert-padding semantics.
- Exact vision ordering for 1, 4, and 16 images plus server-side rejection of
  image 17.
- The exact 69-case tool suite at parallelism 8 with thinking enabled for both
  K3 and K3.25, including every prompt result and per-case comparison.
- An exact 128K prefix replay with a 114,688-token cache-hit delta.
- One cold exact 1,000,000-token request with six successful needles from 50K through 990K.
- A post-1M C16 soak and final-image readiness audit with 30 post-ready API
  requests and zero post-ready JIT warnings.
- ReplaySSM and full-state matched 120-request rolling stresses, the strided
  CUDA materializer, ReplaySSM 128K replay/1M needle, and an explicit measured
  reason for leaving baseline rollback as the DFlash default.

Raw v0.7 machine-readable artifacts live under `benchmarks/v0.7.0-k325/`;
[RESULTS.md](benchmarks/RESULTS.md) defines their timing, scoring, power, and
attribution methods. The v0.6 evidence remains intact in its historical
directory.

The 2026-09-06 tool-eval refresh additionally exercises all 88 standard and
Hard Mode cases using evaluator commit
`cf54b4bfe705f12f71e8866f10730572497c8105` (`2.6.1.dev45+gcf54b4bfe`).
Both targets run in the public v0.7.0 container with matching serving flags
and 400 W/GPU limits, default thinking, and evaluation parallelism 8.
All cases are graded; K3 scores 156/176 and K3.25 scores 164/176. The
original 69-case receipts remain intact, and the new runs, environment
captures, and comparison live under
[`tool-eval-20260906/`](benchmarks/v0.7.0-k325/tool-eval-20260906/).

## Upstream and nearby references

- https://github.com/vllm-project/vllm/pull/52816
- https://github.com/vllm-project/vllm/commit/b389ac29465b33f9e9c534df221ea3c129e9793f
- https://github.com/vllm-project/vllm/pull/53906
- https://github.com/vllm-project/vllm/pull/51540
- https://github.com/vllm-project/vllm/pull/50005
- https://github.com/vllm-project/vllm/issues/53963
- https://github.com/vllm-project/vllm/pull/48792
- https://github.com/vllm-project/vllm/pull/49847
- https://github.com/vllm-project/vllm/pull/49887
- https://github.com/vllm-project/vllm/pull/52928
- https://github.com/vllm-project/vllm/pull/54103
- https://github.com/vllm-project/vllm/pull/54609
- https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell/commit/1755c0f0c01b98463a7b87ab613a6c894b569298
- https://github.com/vllm-project/vllm/pull/49652
- https://github.com/tpurtell/sparkinfer-glmrt
- https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark
- https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell
- https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw

The Mia recipe targets two networked SM121 DGX Sparks; this recipe borrows
relevant SM12x stability lessons but qualifies local PCIe collectives on two
workstation GPUs. The Samuel Cardillo recipe helped flag DFlash as the likely
source of the reported high decode result, which this release then tested
directly on the EXL3 target. Brandon's public recipe supplied the EP2/runtime
optimization lead for the controlled topology comparison. No private
route-128 source, binary, or behavior was used.
