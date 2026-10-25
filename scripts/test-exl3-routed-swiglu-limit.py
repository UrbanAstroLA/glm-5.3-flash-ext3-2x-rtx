#!/usr/bin/env python3
"""CPU regression gate: the EXL3 routed experts' B12x plans receive layer.swiglu_limit (port-exl3-routed-swiglu-limit.py).

Reads the installed exl3.py source (it is not imported: the EXL3 module loads CUDA extensions) and checks that
every B12x Caps(...) and compile_mixed_trellis(...) call passes swiglu_limit. Plumbing only: whether the limit
engages on real activations is not exercised here.

Run inside the recipe image:
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD/scripts:/s:ro" \
    --entrypoint python3 IMAGE /s/test-exl3-routed-swiglu-limit.py [--expect-unpatched] [--output report.json]
--expect-unpatched reproduces the omission on v0.9.1; the default checks the port.
"""
import argparse
import ast
import importlib.util
import json

ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
ap.add_argument("--expect-unpatched", action="store_true")
ap.add_argument("--output", default="")
args = ap.parse_args()
PATCHED = not args.expect_unpatched
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


path = importlib.util.find_spec("vllm.model_executor.layers.quantization.exl3").origin
src = open(path).read()
caps = kwargs_of_calls(src, "Caps")
trellis = kwargs_of_calls(src, "compile_mixed_trellis")
ok_calls = bool(caps) and bool(trellis) and all("swiglu_limit" in k for k in caps + trellis)
check("swiglu-limit: every B12x Caps(...) and compile_mixed_trellis(...) call passes swiglu_limit", ok_calls,
      f"({sum('swiglu_limit' in k for k in caps)}/{len(caps)} Caps, "
      f"{sum('swiglu_limit' in k for k in trellis)}/{len(trellis)} compile_mixed_trellis)")

ok = all(r["met"] for r in RESULTS)
print(f"{'ALL MET' if ok else 'NOT MET'}: {sum(r['met'] for r in RESULTS)}/{len(RESULTS)} "
      f"({'port expected' if PATCHED else 'v0.9.1 omission expected'})")
if args.output:
    json.dump(dict(expect="patched" if PATCHED else "unpatched", results=RESULTS), open(args.output, "w"), indent=1)
raise SystemExit(0 if ok else 1)
