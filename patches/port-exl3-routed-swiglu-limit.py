#!/usr/bin/env python3
"""Pass the model's routed-expert SwiGLU limit to the EXL3 / B12x MoE kernels.

GLM-5.3-Flash sets ``swiglu_limit: 10.0``. FusedMoE stores it on the routed
experts layer (``layer.swiglu_limit``), and dense MLPs and shared experts apply
it, but the EXL3 MoE method plans the B12x kernels without it: the
projection-mixed runtime's ``api.Caps(...)`` and the expert-mixed
``compile_mixed_trellis(...)`` receive no ``swiglu_limit``, so routed-expert
gate/up activations are never clamped. B12x already accepts the limit
(``TPMoEScratchCaps.swiglu_limit``, ``compile_mixed_trellis(swiglu_limit=)``).
Upstream vLLM passes the limit to FusedMoE for this model and has no EXL3 path,
so there is nothing to port; this adds the two keyword arguments. Every layer of
this model uses the same value.
Usage: port-exl3-routed-swiglu-limit.py VLLM_DIR
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


patch('model_executor/layers/quantization/exl3.py', [
    (
"""                force_tile_config=mixed["tile_config"],
                rotation_input_dtype=("bf16" if x.dtype == torch.bfloat16 else "fp16"),
                route_ids_dtype=topk_ids.dtype,
            )
            return {
                "capacity": capacity,
""",
"""                force_tile_config=mixed["tile_config"],
                rotation_input_dtype=("bf16" if x.dtype == torch.bfloat16 else "fp16"),
                route_ids_dtype=topk_ids.dtype,
                swiglu_limit=getattr(layer, "swiglu_limit", None),
            )
            return {
                "capacity": capacity,
""",
        1,
    ),
    (
"""                    if bf16_epilogue
                    else torch.float32
                ),
                **mixed_launch_caps,
            )
            plan = api.plan(caps)
""",
"""                    if bf16_epilogue
                    else torch.float32
                ),
                # Routed-expert SwiGLU clamp from the model config
                # (swiglu_limit); FusedMoE stores it on the layer.
                swiglu_limit=getattr(layer, "swiglu_limit", None),
                **mixed_launch_caps,
            )
            plan = api.plan(caps)
""",
        1,
    ),
], marker='swiglu_limit=getattr(layer, "swiglu_limit", None)')



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

print("EXL3 routed experts receive the model's swiglu_limit")
