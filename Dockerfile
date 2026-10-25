# syntax=docker/dockerfile:1.7

# The four EXL3/B12x vLLM source files come from the qualified vLLM fork
# 30038602b71395f481ef4a6edfe4fcf8551d9c15, as shipped unmodified in
# ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c105...
# They are vendored under vendor/exl3-source (hashes in PROVENANCE.md) so a
# rebuild does not pull that 11 GB image. The runtime is the GLM-5.3 image.
ARG GLM_BASE_IMAGE=cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5

FROM ${GLM_BASE_IMAGE}

ARG B12X_REPOSITORY=https://github.com/tpurtell/sparkinfer-glmrt
ARG B12X_COMMIT=7fcc094edcc93af61fdfbe14300100e3204363ea
ARG DFLASH2_VLLM_COMMIT=b389ac29465b33f9e9c534df221ea3c129e9793f

SHELL ["/bin/bash", "-c"]

# Fetch an immutable snapshot of the user's current B12x fork. Keep the GLM
# base's Torch 2.13, CUDA 13, and CUTLASS DSL 4.6.2 stack intact; B12x is pure
# Python/CuTe DSL and compiles its selected SM120 specializations at runtime.
RUN B12X_REPOSITORY="${B12X_REPOSITORY}" B12X_COMMIT="${B12X_COMMIT}" \
    python3 - <<'PY'
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

repository = os.environ["B12X_REPOSITORY"].removesuffix(".git")
commit = os.environ["B12X_COMMIT"]
archive = Path("/tmp/b12x.tar.gz")
urllib.request.urlretrieve(f"{repository}/archive/{commit}.tar.gz", archive)
with tarfile.open(archive) as tar:
    tar.extractall("/tmp", filter="data")
sources = list(Path("/tmp").glob("sparkinfer-glmrt-*"))
if len(sources) != 1:
    raise RuntimeError(f"expected one B12x source tree, found: {sources}")
shutil.rmtree("/opt/b12x", ignore_errors=True)
shutil.move(sources[0], "/opt/b12x")
archive.unlink()
PY
RUN python3 -m pip install --no-cache-dir --no-deps -e /opt/b12x
# The fork registers vLLM general plugins (FP6, direct-I/O loader) written for
# a newer vLLM than this GLM base. The recipe uses neither; load only vLLM's
# own LoRA resolvers so those entry points never import.
ENV VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver

# Carry only the proven EXL3 quantization implementation into the GLM vLLM
# tree, then adapt its narrow registration/model-recognition surface.
COPY vendor/exl3-source/vllm/model_executor/layers/quantization/exl3.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
# Reuse the mature vLLM/B12x adapter from the same qualified source image. The
# adapter is ported below onto the newer GLM-5.3 vLLM APIs; all kernels still
# resolve from the current /opt/b12x checkout pinned above.
COPY vendor/exl3-source/vllm/v1/attention/backends/mla/b12x_mla_sparse.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
COPY vendor/exl3-source/vllm/model_executor/layers/sparse_attn_indexer.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py
COPY vendor/exl3-source/vllm/model_executor/layers/mla_cache_format.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mla_cache_format.py
COPY patches/port-exl3-glm53.py /tmp/port-exl3-glm53.py
COPY patches/port-exl3-ep-glm53.py /tmp/port-exl3-ep-glm53.py
COPY patches/port-exl3-mtp-glm53.py /tmp/port-exl3-mtp-glm53.py
COPY patches/port-exl3-projection-mixed-glm53.py /tmp/port-exl3-projection-mixed-glm53.py
COPY patches/port-exl3-prepared-dtype.py /tmp/port-exl3-prepared-dtype.py
COPY patches/port-b12x-glm53.py /tmp/port-b12x-glm53.py
COPY patches/port-b12x-kpool-glm53.py /tmp/port-b12x-kpool-glm53.py
COPY patches/b12x_dcp_topk.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_dcp_topk.py
COPY patches/port-b12x-dcp-glm53.py /tmp/port-b12x-dcp-glm53.py
COPY patches/port-b12x-dcp-owner-glm53.py /tmp/port-b12x-dcp-owner-glm53.py
COPY patches/port-b12x-nvfp4-glm53.py /tmp/port-b12x-nvfp4-glm53.py
COPY patches/port-b12x-glm-h64-query.py /tmp/port-b12x-glm-h64-query.py
COPY patches/port-b12x-mhc-glm53.py /tmp/port-b12x-mhc-glm53.py
COPY patches/port-glm53-mhc-warmup.py /tmp/port-glm53-mhc-warmup.py
COPY patches/port-glm53-sm12-stability.py /tmp/port-glm53-sm12-stability.py
COPY patches/port-b12x-glm-index-width.py /tmp/port-b12x-glm-index-width.py
COPY patches/port-glm-sparse-memory.py /tmp/port-glm-sparse-memory.py
COPY patches/port-glm-prefill-jit.py /tmp/port-glm-prefill-jit.py
COPY patches/b12x_pcie_all_reduce.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/b12x_pcie_all_reduce.py
COPY patches/port-b12x-pcie-glm53.py /tmp/port-b12x-pcie-glm53.py
COPY patches/b12x_dcp_a2a.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/b12x_dcp_a2a.py
COPY patches/port-b12x-dcp-a2a-glm53.py /tmp/port-b12x-dcp-a2a-glm53.py
COPY patches/vllm-replayssm-spec.patch /tmp/vllm-replayssm-spec.patch
COPY patches/vllm-dynamic-sd-cudagraph.patch /tmp/vllm-dynamic-sd-cudagraph.patch
COPY patches/port-replayssm-glm53.py /tmp/port-replayssm-glm53.py
COPY patches/port-glm-replayssm-conv-window.py /tmp/port-glm-replayssm-conv-window.py
COPY patches/port-glm53-mtp-prefix-cache.py /tmp/port-glm53-mtp-prefix-cache.py
COPY patches/adaptive_mtp.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dynamic/adaptive_mtp.py
COPY patches/port-adaptive-mtp-glm53.py /tmp/port-adaptive-mtp-glm53.py
COPY patches/port-dflash2-glm53.py /tmp/port-dflash2-glm53.py
COPY patches/port-dflash2-glm-eagle3.py /tmp/port-dflash2-glm-eagle3.py
COPY patches/port-dflash2-glm-kv.py /tmp/port-dflash2-glm-kv.py
COPY patches/port-dflash2-replicated-dcp.py /tmp/port-dflash2-replicated-dcp.py
COPY patches/port-b12x-latest-api.py /tmp/port-b12x-latest-api.py
COPY patches/port-glm53-layer-owner.py /tmp/port-glm53-layer-owner.py
COPY patches/port-glm53-graph-memory.py /tmp/port-glm53-graph-memory.py
COPY patches/port-glm53-draft-slots.py /tmp/port-glm53-draft-slots.py
COPY patches/port-glm53-breakable-capture-sync.py /tmp/port-glm53-breakable-capture-sync.py
COPY patches/port-b12x-glm-next-records.py /tmp/port-b12x-glm-next-records.py
COPY patches/port-kpool-seed-stride-glm53.py /tmp/port-kpool-seed-stride-glm53.py
COPY patches/port-kpool-spec-ring-glm53.py /tmp/port-kpool-spec-ring-glm53.py
COPY patches/port-dflash2-boundary-prefix-cache.py /tmp/port-dflash2-boundary-prefix-cache.py
COPY patches/port-dcp1-kpool-tail-glm53.py /tmp/port-dcp1-kpool-tail-glm53.py
COPY patches/port-kpool-tail-positions-glm53.py /tmp/port-kpool-tail-positions-glm53.py
COPY patches/port-kpool-unaligned-chunk-glm53.py /tmp/port-kpool-unaligned-chunk-glm53.py
COPY patches/port-kpool-col511-lowest-glm53.py /tmp/port-kpool-col511-lowest-glm53.py
COPY patches/port-router-fp32-out-sm12x.py /tmp/port-router-fp32-out-sm12x.py
COPY patches/port-exl3-routed-swiglu-limit.py /tmp/port-exl3-routed-swiglu-limit.py
RUN python3 /tmp/port-exl3-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-exl3-ep-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-exl3-mtp-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-exl3-projection-mixed-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-exl3-prepared-dtype.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-kpool-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-dcp-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-dcp-owner-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-nvfp4-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-glm-h64-query.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-mhc-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-mhc-warmup.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-sm12-stability.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-glm-index-width.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm-sparse-memory.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm-prefill-jit.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-pcie-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-dcp-a2a-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm

# Port the current upstream ReplaySSM series onto the exact day-zero vLLM
# commit, then add compact rollback for GLM's vector-gated KDA recurrence. The
# patch is intentionally applied after the local EXL3/B12x ports; this is the
# ordering qualified by the clean-image compatibility test.
RUN cd /usr/local/lib/python3.12/dist-packages \
 && patch --batch --forward -p1 < /tmp/vllm-replayssm-spec.patch \
 && patch --batch --forward -p1 < /tmp/vllm-dynamic-sd-cudagraph.patch \
 && python3 /tmp/port-replayssm-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm-replayssm-conv-window.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-mtp-prefix-cache.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-adaptive-mtp-glm53.py \
    /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm

# DFlash2 landed after the GLM day-zero branch diverged from vLLM main. Apply
# that immutable upstream runtime delta after the local ReplaySSM/adaptive-MTP
# series, which is the exact source shape qualified below. The port script
# excludes tests and resolves two registration-only branch conflicts, then
# guards the decoder-layer indirection fixed upstream after vLLM #53428.
RUN DFLASH2_VLLM_COMMIT="${DFLASH2_VLLM_COMMIT}" python3 - <<'PY'
import os
import urllib.request

commit = os.environ["DFLASH2_VLLM_COMMIT"]
urllib.request.urlretrieve(
    f"https://github.com/vllm-project/vllm/commit/{commit}.patch",
    "/tmp/vllm-dflash2.patch",
)
PY
RUN python3 /tmp/port-dflash2-glm53.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
      /tmp/vllm-dflash2.patch \
 && python3 /tmp/port-dflash2-glm-eagle3.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-dflash2-glm-kv.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-dflash2-replicated-dcp.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-latest-api.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-layer-owner.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-graph-memory.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-draft-slots.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-glm53-breakable-capture-sync.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-b12x-glm-next-records.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-kpool-seed-stride-glm53.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-kpool-spec-ring-glm53.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-dflash2-boundary-prefix-cache.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-dcp1-kpool-tail-glm53.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-kpool-tail-positions-glm53.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-kpool-unaligned-chunk-glm53.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-kpool-col511-lowest-glm53.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-router-fp32-out-sm12x.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-exl3-routed-swiglu-limit.py \
      /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 -m compileall -q /usr/local/lib/python3.12/dist-packages/vllm

ENV PYTHONPATH=/opt/b12x:/usr/local/lib/python3.12/dist-packages \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_MODULE_LOADING=LAZY \
    VLLM_EXL3_TRELLIS_MIN_M=1 \
    VLLM_EXL3_TRELLIS_MAX_M=32 \
    VLLM_EXL3_TRELLIS_BLOCK_M=8 \
    VLLM_EXL3_PREFILL_TRELLIS=1 \
    VLLM_EXL3_PREFILL_BLOCK_M=64 \
    VLLM_EXL3_PREFILL_CAPACITY=1024 \
    VLLM_B12X_GLM_H64_QUERY_PROJ=auto \
    VLLM_USE_B12X_MHC=auto \
    VLLM_USE_B12X_SPARSE_INDEXER=1

# Build-time compatibility and scope probe. This proves that the newer vLLM
# imports the transplanted method, that the current B12x API is present, and
# that GLM enters B12x standard-fused mode without inventing FP8 base weights.
RUN python3 - <<'PY'
from pathlib import Path
from types import SimpleNamespace

import b12x
import cutlass
import torch
import vllm
from b12x.moe import ep_moe, fused_moe
from b12x.moe.fused_moe._impl import _projection_mixed_route_map
from b12x.moe.fused_moe._tuning import MoeDecodeConfig
from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
    build_projection_tiered_maps,
)
from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights
from b12x.attention import dsa_indexer, sparse_mla
from b12x.attention.dsa_indexer.paged import index_topk_fp8
from b12x.attention.dsa_indexer.scratch import (
    INDEXER_SOURCE_LAYOUT_PAGED,
    B12XIndexerScratchCaps,
    plan_indexer_scratch,
)
from b12x.gemm import bf16_gemv, mla_query_projection
from b12x.preparation import PreparedCall, prepare_default
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    _exl3_moe_weight_loader,
    _load_b12x_fused_moe,
)
from vllm.model_executor.layers.b12x_preparation import (
    prepare_b12x_plan,
    retain_b12x_programs,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3DecoderLayer,
    DFlash2Qwen3Model,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.interfaces import supports_eagle3, supports_replayssm
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)
from vllm.third_party.flash_linear_attention.ops.kda_replayssm_spec_decode import (
    kda_replayssm_spec_decode,
    materialize_kda_replayssm_state,
)
from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.spec_decode.dynamic.adaptive_mtp import AdaptiveMTPController
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum

sparse_indexer_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "sparse_attn_indexer.py"
).read_text()
assert "b12x.attention.nsa_indexer" not in sparse_indexer_source
assert sparse_indexer_source.count("b12x.attention.dsa_indexer") == 9
assert "SOURCE_LAYOUT_PAGED as INDEXER_SOURCE_LAYOUT_PAGED" not in sparse_indexer_source
assert "_b12x_dcp_topk_owner_stage(" in sparse_indexer_source

# ReplaySSM's request-index metadata must remain safe for both rolling mixed
# batches and non-contiguous block-table columns.  The former matches the
# intermittent C4 failure reproduced by Samuel Cardillo's derivative; the
# latter is an adjacent KDA failure mode documented by upstream vLLM work.
cudagraph_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/"
    "cudagraph_utils.py"
).read_text()
assert (
    "mixed_mode\n                and not "
    "self.vllm_config.cache_config.use_replayssm_spec"
) in cudagraph_source
kda_replayssm_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/third_party/"
    "flash_linear_attention/ops/kda_replayssm_spec_decode.py"
).read_text()
for stride_contract in (
    "state_indices.stride(0)",
    "source_indices.stride(0)",
    "has_initial_state.stride(0)",
    "row_write_pos.stride(0)",
    "row_cache_base.stride(0)",
):
    assert stride_contract in kda_replayssm_source
assert "ReplaySSM prefill row count mismatch" in kda_replayssm_source
glm_kda_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py"
).read_text()
assert "self.num_spec + 1" in glm_kda_source

def entry(name: str, bits: int):
    return {
        "bits_per_weight": bits,
        "quant_format": "exl3",
        "stored_tensors": {
            f"{name}.suh": {},
            f"{name}.svh": {},
            f"{name}.trellis": {},
            f"{name}.mcg": {},
        },
    }

storage = {}
for expert_id, rates in enumerate(((3, 4, 3), (4, 3, 4))):
    root = f"model.language_model.layers.3.mlp.experts.{expert_id}"
    for projection, bits in zip(
        ("gate_proj", "up_proj", "down_proj"), rates, strict=True
    ):
        name = f"{root}.{projection}"
        storage[name] = entry(name, bits)
for expert_id in range(2):
    root = f"mtp.0.mlp.experts.{expert_id}"
    for projection in ("gate_proj", "up_proj", "down_proj"):
        name = f"{root}.{projection}"
        storage[name] = entry(name, 3)
config = Exl3Config(bits=3.25, codebook="mcg", tensor_storage=storage)
config._configure_standard_fused_moe(
    SimpleNamespace(model_type="glm5_next", num_hidden_layers=45)
)
config._configure_base_quantization(SimpleNamespace(model_type="glm5_next"))

assert get_quantization_config("exl3") is Exl3Config
assert supports_replayssm(Glm5NextForConditionalGeneration)
assert MambaStateShapeCalculator.replayssm_spec_ring_len(10, 5) == 16
assert MambaStateShapeCalculator.replayssm_spec_ring_len(16, 5) == 32
assert callable(kda_replayssm_spec_decode)
assert callable(materialize_kda_replayssm_state)
assert supports_eagle3(Glm5NextForCausalLM)
assert supports_eagle3(Glm5NextForConditionalGeneration)
assert "DFlash2DraftModel" in ModelRegistry.get_supported_archs()
assert DFlash2Qwen3Model.decoder_layer_cls is DFlash2Qwen3DecoderLayer
assert "self.decoder_layer_cls(" in Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/"
    "qwen3_dflash.py"
).read_text()
replicated_dflash_port = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py"
).read_text()
assert "dcp_shard_count_override" in replicated_dflash_port
assert "sw_block_size = 128" in Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "attention/attention.py"
).read_text()
adaptive_probe = AdaptiveMTPController(max_depth=5, probe_interval=4)
assert adaptive_probe.select(["c1"], 1, 5) == 5
assert adaptive_probe.select(["c8"], 8, 1) == 1
assert config.standard_fused_moe
assert config.standard_layer_projection_bitrates(
    "model.layers.3.mlp", 2
) == ((3, 4, 3), (4, 3, 4))
assert config.standard_layer_projection_bitrates(
    "model.layers.45.mlp", 2
) == ((3, 3, 3), (3, 3, 3))
assert config._base_quant_config is None
assert config._moe_prefix_is_exl3(
    "language_model.model.layers.3.mlp.experts"
)
assert config._moe_prefix_is_exl3("model.layers.3.mlp.experts")
assert config._storage_entry(
    "model.layers.3.mtp_block.mlp.experts.0.gate_proj"
) is not None
assert config._moe_prefix_is_exl3(
    "model.layers.3.mtp_block.mlp.experts.routed_experts"
)
assert "is_standard_mtp" in Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/exl3.py"
).read_text()
assert "scheduler_config.max_num_seqs" in Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/exl3.py"
).read_text()
# Current B12x removed the vLLM fused-MoE compatibility shim; the ported
# EXL3 adapter binds the scratch-planned _impl layer through the same names.
assert callable(fused_moe.plan_execution)
assert callable(fused_moe.bind)
assert callable(fused_moe.run)
exl3_moe = _load_b12x_fused_moe()
for exl3_moe_entry in (
    "plan_weights", "prepare_weights", "Caps", "plan", "bind", "run",
    "required_nbytes",
):
    assert callable(getattr(exl3_moe, exl3_moe_entry)), exl3_moe_entry
assert ProjectionTrellisTierWeights is not None
projection_plan = exl3_moe.plan_weights(
    quant_modes="w4a16",
    source_format="exl3_trellis_mcg",
    activation="silu",
    params_dtype=torch.bfloat16,
    num_experts=144,
    hidden_size=128,
    intermediate_size=128,
    w13_layout="trellis_t256_proj",
    w4a16_layout="trellis_native",
    trellis_bits=3,
    trellis_codebook="mcg",
    trellis_rate_granularity="per_expert_projection",
)
assert projection_plan.trellis_tile_config is None
projection_caps = exl3_moe.Caps(
    max_tokens=1,
    num_topk=6,
    route_num_experts=288,
    device="cpu",
    weight_plan=projection_plan,
    quant_mode="w4a16",
    full_rotation_output_dtype=torch.bfloat16,
)
assert projection_caps.decode_config == MoeDecodeConfig(
    backend="w4a16",
    route_planner="internal",
    max_active_clusters=None,
    w4a16_route_mode="packed",
)
assert exl3_moe.required_nbytes(projection_caps) > 0
projection_scratch_plan = exl3_moe.plan(projection_caps)
assert (
    projection_scratch_plan._core_workspace_plan.full_rotation_output_dtype
    == torch.bfloat16
)
glm_projection_plan = exl3_moe.plan_weights(
    quant_modes="w4a16",
    source_format="exl3_trellis_mcg",
    activation="silu",
    params_dtype=torch.bfloat16,
    num_experts=288,
    hidden_size=128,
    intermediate_size=128,
    w13_layout="trellis_t256_proj",
    w4a16_layout="trellis_native",
    trellis_bits=3,
    trellis_codebook="mcg",
    trellis_rate_granularity="per_expert_projection",
)
assert exl3_moe.required_nbytes(exl3_moe.Caps(
    max_tokens=1,
    num_topk=8,
    route_num_experts=288,
    device="cpu",
    weight_plan=glm_projection_plan,
    quant_mode="w4a16",
)) > 0
_, glm_descriptor = build_projection_tiered_maps(
    [0] * 288,
    [0] * 288,
    [0] * 288,
    tier_slots=(288, 0),
    device=torch.device("cpu"),
)
assert glm_descriptor.view(3, 288)[:, 287].tolist() == [287, 287, 287]
precomposed_map = torch.tensor(
    list(range(144)) + [-1] * 144, dtype=torch.int32
)
assert _projection_mixed_route_map(
    torch.arange(144, dtype=torch.int32),
    precomposed_map,
    route_num_experts=288,
    device=torch.device("cpu"),
).data_ptr() == precomposed_map.data_ptr()
trellis_plan = exl3_moe.plan_weights(
    quant_modes="w4a16",
    source_format="b12x_trellis",
    activation="silu",
    params_dtype=torch.bfloat16,
    num_experts=2,
    hidden_size=128,
    intermediate_size=128,
    trellis_bits=3,
    trellis_codebook="mcg",
    trellis_tile_config=(64, 128, 64, 128),
)
assert trellis_plan.source_format == "b12x_trellis"
assert trellis_plan.trellis_codebook == "mcg"
assert trellis_plan.io_dtype == "bfloat16"
# Uniform Trellis EP now uses the fused route-map path (current ep_moe no
# longer compiles full-rotation Trellis).
uniform_ep_caps = exl3_moe.Caps(
    max_tokens=1,
    num_topk=1,
    route_num_experts=4,
    device="cpu",
    weight_plan=trellis_plan,
    quant_mode="w4a16",
)
assert exl3_moe.required_nbytes(uniform_ep_caps) > 0
assert callable(ep_moe.prepare_expert_map)
assert callable(ep_moe.plan)
assert callable(ep_moe.bind)
assert callable(ep_moe.run)
rank1_map = tuple([-1] * 144 + list(range(144)))
prepared_rank1_map = ep_moe.prepare_expert_map(
    torch.tensor(rank1_map, dtype=torch.int32),
    local_num_experts=144,
    global_num_experts=288,
)
assert prepared_rank1_map.tensor[143].item() == -1
assert prepared_rank1_map.tensor[144].item() == 0
assert prepared_rank1_map.tensor[287].item() == 143
loaded_local_ids = []
fake_parameter = SimpleNamespace(
    map_global_expert_id=lambda expert_id: rank1_map[expert_id],
    load_exl3_weight=lambda _weight, *, expert_id, shard_id: (
        loaded_local_ids.append((expert_id, shard_id))
    ),
)
assert not _exl3_moe_weight_loader(
    fake_parameter, torch.ones(1), "nonlocal", "w1", 143, return_success=True
)
assert _exl3_moe_weight_loader(
    fake_parameter, torch.ones(1), "local", "w1", 144, return_success=True
)
assert loaded_local_ids == [(0, "w1")]
exl3_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/exl3.py"
).read_text()
uniform_start = exl3_source.index("    def _prepare_rank_sliced_weights(")
uniform_end = exl3_source.index(
    "    def get_fused_moe_quant_config(", uniform_start
)
assert exl3_source[uniform_start:uniform_end].count(
    "params_dtype=layer.exl3_params_dtype,"
) == 2
assert "local_expert_id = param.map_global_expert_id(expert_id)" in exl3_source
assert "if local_expert_id < 0:" in exl3_source
assert "layer.exl3_prepared_ep_map" not in exl3_source
assert 'kwargs["route_expert_map"] = layer.exl3_projection_route_map' in exl3_source
assert "b12x-latest-api: scratch-planned fused MoE" in exl3_source
assert "bf16_gemv.mm(x_2d, weight, plan=plan)" in exl3_source
assert 'source_format="b12x_trellis"' in exl3_source
assert 'trellis_codebook="mcg"' in exl3_source
assert "marker_value = int(marker.item()) & 0xFFFFFFFF" in exl3_source
assert 'source_format="exl3_trellis_mcg"' in exl3_source
assert callable(sparse_mla.plan)
assert callable(sparse_mla.bind)
assert callable(sparse_mla.run)
assert "run_decode" not in sparse_mla.META.entry_points
glm_mla_caps = sparse_mla.Caps(
    device="cpu",
    num_q_heads=64,
    max_q_rows=96,
    max_width=2176,
    softmax_scale=576**-0.5,
    kv_dtype=torch.uint8,
    head_dim=576,
    v_head_dim=512,
    model_type=int(sparse_mla.ModelType.GLM_NSA),
    cache_record_bytes=656,
    mode="decode",
    max_chunks_per_row=34,
    page_size=64,
    head_major_output=True,
    return_lse=True,
    lse_scale="natural",
)
assert glm_mla_caps.cache_record_bytes == 656
assert glm_mla_caps.scale_format == 1 and not glm_mla_caps.fp8_rope
assert sparse_mla.plan(glm_mla_caps).component_id == "attention.sparse_mla"
assert callable(dsa_indexer.plan)
assert callable(index_topk_fp8)
assert INDEXER_SOURCE_LAYOUT_PAGED == "paged"
assert plan_indexer_scratch(B12XIndexerScratchCaps(
    device="cpu",
    source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
    num_q_heads=32,
    max_q_rows=8,
    max_page_table_width=34,
    topk=2176,
    mode="decode",
    shared_page_table=False,
)).shapes_and_dtypes()
assert callable(prepare_default) and PreparedCall is not None
assert callable(prepare_b12x_plan) and callable(retain_b12x_programs)
assert callable(bf16_gemv.plan) and callable(bf16_gemv.query_from_call)
assert "plan" in __import__("inspect").signature(bf16_gemv.mm).parameters
assert callable(mla_query_projection.run_glm_h64_bf16)
from b12x.norm import mhc
assert callable(mhc.run_post_pre)
assert callable(mhc.plan)
assert "plan" in __import__("inspect").signature(mhc.run_post_pre).parameters
assert "nope" in __import__("inspect").signature(
    mla_query_projection.prewarm_glm_h64_bf16
).parameters
from b12x.comm.pcie import OneshotAllReducePool
from b12x.comm.pcie import DcpTopKOwnerExchange
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
    B12xPcieAllReduce,
)
from b12x.comm.pcie import plan as pcie_plan
from b12x.comm.pcie import query_from_runtime as pcie_query_from_runtime
assert callable(pcie_plan) and callable(pcie_query_from_runtime)
assert "plan" in __import__("inspect").signature(
    OneshotAllReducePool.all_reduce
).parameters
assert "plan" in __import__("inspect").signature(
    PCIeDCPA2APool.lse_reduce_scatter
).parameters
assert callable(OneshotAllReducePool.from_exchange_group)
assert callable(DcpTopKOwnerExchange.from_exchange_group)
assert callable(PCIeDCPA2APool.from_exchange_group)
assert B12xPcieAllReduce.backend_name(None) == "B12X_PCIE_ONESHOT"
assert B12xMLASparseBackend.get_supported_head_sizes() == [512, 576]
sparse_backend_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
    "b12x_mla_sparse.py"
).read_text()
assert "int(layer.impl.topk_tokens)" in sparse_backend_source
assert "_prepare_b12x_sparse_mla_plan(" in sparse_backend_source
assert "run_decode" not in sparse_backend_source
assert "lse = lse_base2 * _LN2" in sparse_backend_source
pcie_adapter_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/"
    "device_communicators/b12x_pcie_all_reduce.py"
).read_text()
assert "self.runtime.all_reduce(inp, plan=self._plan_for(inp))" in pcie_adapter_source
dcp_a2a_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/"
    "device_communicators/b12x_dcp_a2a.py"
).read_text()
assert "plan=self.gather_plan" in dcp_a2a_source
assert "plan=self.combine_plan" in dcp_a2a_source
assert "self.pool.prepare_graph_" not in dcp_a2a_source
indexer_backend_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
    "indexer.py"
).read_text()
assert "scheduler_config.max_num_seqs" in indexer_backend_source
mla_attention_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "attention/mla_attention.py"
).read_text()
assert 'self.attn_backend.get_name() == "B12X_MLA_SPARSE"' in mla_attention_source
assert CacheConfig(cache_dtype="nvfp4_ds_mla").cache_dtype == "nvfp4_ds_mla"
assert CacheConfig(cache_dtype="fp8_ds_mla").cache_dtype == "fp8_ds_mla"
assert STR_DTYPE_TO_TORCH_DTYPE["nvfp4_ds_mla"] is torch.uint8
assert STR_DTYPE_TO_TORCH_DTYPE["fp8_ds_mla"] is torch.uint8
assert B12xMLASparseBackend.get_kv_cache_shape(1, 64, 1, 576, "nvfp4_ds_mla") == (
    1,
    64,
    432,
)
assert MLAAttentionSpec(
    block_size=64,
    num_kv_heads=1,
    head_size=576,
    dtype=torch.uint8,
    cache_dtype_str="nvfp4_ds_mla",
).real_page_size_bytes == 64 * 432
assert B12xMLASparseBackend.get_kv_cache_shape(1, 64, 1, 576, "fp8_ds_mla") == (
    1,
    64,
    656,
)
# GLM_NEXT 528-byte records are opt-in: the default build keeps 656 bytes for
# GLM's 512-wide MLA spec, and VLLM_GLM53_NOPE_RECORD=1 switches every layer.
assert B12xMLASparseBackend.get_kv_cache_shape(1, 64, 1, 512, "fp8_ds_mla") == (
    1,
    64,
    656,
)
from vllm.model_executor.layers.mla_cache_format import (
    GLM53_NOPE_RECORD,
    NVFP4_MLA_CACHE_FORMAT,
)
assert not GLM53_NOPE_RECORD
assert NVFP4_MLA_CACHE_FORMAT.record_abi("fp8_ds_mla") == "vllm-default-v1"
import subprocess
import sys
glm_next_probe = subprocess.run(
    [sys.executable, "-c", '''
import torch
from b12x.attention import sparse_mla
from vllm.config.cache import CacheConfig
from vllm.model_executor.layers.mla_cache_format import (
    GLM53_NOPE_RECORD,
    NVFP4_MLA_CACHE_FORMAT,
)
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseBackend
from vllm.v1.kv_cache_interface import MLAAttentionSpec

assert GLM53_NOPE_RECORD
assert NVFP4_MLA_CACHE_FORMAT.record_abi("fp8_ds_mla") == (
    "fp8_ds_mla:glm-next-nope-528:v1"
)
assert B12xMLASparseBackend.get_kv_cache_shape(1, 64, 1, 512, "fp8_ds_mla") == (
    1, 64, 528,
)
assert B12xMLASparseBackend.get_kv_cache_shape(1, 64, 1, 576, "fp8_ds_mla") == (
    1, 64, 656,
)
assert MLAAttentionSpec(
    block_size=4352, num_kv_heads=1, head_size=512, dtype=torch.uint8,
    cache_dtype_str="fp8_ds_mla",
).real_page_size_bytes == 4352 * 528
caps = sparse_mla.Caps(
    device="cpu", num_q_heads=64, max_q_rows=96, max_width=2176,
    softmax_scale=256**-0.5, kv_dtype=torch.uint8, head_dim=512,
    v_head_dim=512, model_type=int(sparse_mla.ModelType.GLM_NEXT),
    cache_record_bytes=528, mode="decode", max_chunks_per_row=34,
    page_size=64, head_major_output=True,
)
assert caps.cache_record_bytes == 528 and caps.scale_format == 1
assert not caps.fp8_rope
assert sparse_mla.plan(caps).component_id == "attention.sparse_mla"
assert callable(sparse_mla.plan_cache_writer)
assert callable(sparse_mla.concat_and_cache_glm_next_mla)
assert CacheConfig(cache_dtype="fp8_ds_mla").compute_hash() != ""
print("GLM_NEXT record probe passed")
'''],
    env={**__import__("os").environ, "VLLM_GLM53_NOPE_RECORD": "1"},
    capture_output=True,
    text=True,
)
assert glm_next_probe.returncode == 0, glm_next_probe.stderr[-4000:]
assert "GLM_NEXT record probe passed" in glm_next_probe.stdout
glm_next_backend_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
    "b12x_mla_sparse.py"
).read_text()
assert "concat_and_cache_glm_next_mla(" in glm_next_backend_source
assert "VLLM_GLM53_NOPE_RECORD=1 requires DCP=1" in glm_next_backend_source
assert "glm53-nope-record" in Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "attention/mla_attention.py"
).read_text()
assert Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/"
    "placement.py"
).is_file()
assert "_glm53_slot_shared_draft_specs" in Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
).read_text()
assert MLAAttentionSpec(
    block_size=64,
    num_kv_heads=1,
    head_size=576,
    dtype=torch.uint8,
    cache_dtype_str="fp8_ds_mla",
).real_page_size_bytes == 64 * 656
assert AttentionBackendEnum.B12X_MLA_SPARSE.get_class() is B12xMLASparseBackend
assert "output_physical_slots" in __import__("inspect").signature(
    SparseAttnIndexer.__init__
).parameters
kpool_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "sparse_attn_indexer_kpool.py"
).read_text()
assert "Using B12x fused paged score+top-k for GLM kpool decode." in kpool_source
assert "Using B12x PCIe DCP owner top-k exchange" in __import__(
    "inspect"
).getsource(__import__(
    "vllm.model_executor.layers.sparse_attn_indexer",
    fromlist=["_get_b12x_dcp_topk_owner_exchange"],
))
config_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py"
).read_text()
assert "b12x_glm_nvfp4_mla" in config_source
assert "and not b12x_glm_nvfp4_mla" in config_source
mhc_source = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py"
).read_text()
assert "Using B12x fused GLM H4096 mHC post+pre for decode M=1." in mhc_source
assert "plan=b12x_mhc_plan," in mhc_source
assert Path(b12x.__file__).is_relative_to(Path("/opt/b12x")), b12x.__file__
assert torch.__version__.startswith("2.13."), torch.__version__
assert vllm.__version__ == "0.1.dev20051+g487ecf187", vllm.__version__
assert cutlass.__version__ == "4.6.2", cutlass.__version__
print("GLM-5.3 vLLM + EXL3 + B12x compatibility probe passed")
PY

LABEL org.opencontainers.image.source="https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx" \
      org.opencontainers.image.description="GLM-5.3 Flash EXL3 K3/K4/K3.25 on 2x SM120: DFlash2, multimodal GLM, FP8 MLA, DCP2, and B12x PCIe kernels" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.tpurtell.b12x.source="https://github.com/tpurtell/sparkinfer-glmrt" \
      io.tpurtell.glm-base.digest="sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5" \
      io.tpurtell.exl3-source.digest="sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32" \
      io.tpurtell.exl3-vllm.commit="30038602b71395f481ef4a6edfe4fcf8551d9c15" \
      io.tpurtell.dflash2-vllm.commit="b389ac29465b33f9e9c534df221ea3c129e9793f" \
      io.tpurtell.b12x.commit="${B12X_COMMIT}" \
      io.tpurtell.replayssm.mixed-graph-fix="c51c3856f7f8ba50af3b3a60ff48e7d6a1fa303c"

COPY templates/glm53-zai-a5b45eb.jinja /opt/glm53/templates/glm53-zai-a5b45eb.jinja
COPY container/glm53-entrypoint.sh /usr/local/bin/glm53-entrypoint
COPY container/glm53-release-warmup.py /usr/local/bin/glm53-release-warmup.py
RUN chmod 0755 /usr/local/bin/glm53-entrypoint

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=5 \
  CMD ["python3", "-c", "import pathlib,urllib.request; assert pathlib.Path('/tmp/glm53-release-ready').is_file(); urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()"]

ENTRYPOINT ["/usr/local/bin/glm53-entrypoint"]

# Narrow structured-output corrections; keep the qualified numerical stack.
COPY patches/port-xgrammar-termination.py /tmp/port-xgrammar-termination.py
COPY patches/port-structured-output-reasoning.py /tmp/port-structured-output-reasoning.py
RUN python3 /tmp/port-xgrammar-termination.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-structured-output-reasoning.py /usr/local/lib/python3.12/dist-packages/vllm
LABEL io.tpurtell.structured-output.reasoning-fix="c6e19b3be243" \
      io.tpurtell.structured-output.termination-fix="vllm-pr-52805"
COPY scripts/test-xgrammar-termination.py scripts/test-structured-output-reasoning.py /opt/glm53-tests/
RUN python3 /opt/glm53-tests/test-xgrammar-termination.py \
 && python3 /opt/glm53-tests/test-structured-output-reasoning.py
