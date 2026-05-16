"""
tools/quantize_reranker.py
──────────────────────────
Run ONCE to export a true INT8 ONNX reranker to disk.
After this, every startup loads in ~3s instead of tracing for ~60s.

Usage:
    python tools/quantize_reranker.py

Output:
    ./models/bge-reranker-v2-m3-int8/   (set RERANKER_INT8_PATH to this)

Time: ~3-5 min on i5-1240p (one-time cost)
Disk: ~1.1 GB (INT8) vs 2.2 GB (fp32)

What this actually does (vs the broken export=True path):
  OLD (broken): ORTModelForSequenceClassification.from_pretrained(export=True)
      → exports fp32 ONNX only, no quantization, misleadingly fast to call
  NEW (this):   fp32 ONNX export → ORTQuantizer with QuantizationConfig(AVX512_VNNI)
      → true INT8 weights, 2-3x faster inference on i5-1240p

After running this script, set in your .env or shell:
    RERANKER_INT8_PATH=./models/bge-reranker-v2-m3-int8
"""

import os
import sys
import time
from pathlib import Path

BASE_MODEL  = os.getenv("RERANKER_MODEL",     "BAAI/bge-reranker-v2-m3")
FP32_DIR    = Path("./models/bge-reranker-v2-m3-fp32-onnx")
INT8_DIR    = Path("./models/bge-reranker-v2-m3-int8")
FILE_NAME   = "model.onnx"


def main():
    print("=" * 60)
    print("  Reranker INT8 quantization (one-time setup)")
    print("=" * 60)

    # ── Step 1: Check deps ────────────────────────────────────────
    try:
        from optimum.onnxruntime import (
            ORTModelForSequenceClassification,
            ORTQuantizer,
        )
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        print("\n[ERROR] Missing dependencies. Run:")
        print("  pip install optimum[onnxruntime] onnxruntime")
        sys.exit(1)

    # ── Step 2: Export fp32 ONNX (if not already done) ───────────
    if not (FP32_DIR / FILE_NAME).exists():
        print(f"\n[1/2] Exporting fp32 ONNX from {BASE_MODEL}...")
        print("      (this downloads ~2.2 GB and takes 2-4 min)")
        t0 = time.time()
        FP32_DIR.mkdir(parents=True, exist_ok=True)
        model = ORTModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            export=True,
            provider="CPUExecutionProvider",
        )
        model.save_pretrained(str(FP32_DIR))
        print(f"      Done in {time.time()-t0:.0f}s → {FP32_DIR}")
    else:
        print(f"\n[1/2] fp32 ONNX already at {FP32_DIR} — skipping export")

    # ── Step 3: Quantize to INT8 ──────────────────────────────────
    print(f"\n[2/2] Quantizing to INT8...")
    print("      Strategy: AVX512_VNNI (optimal for Intel i5-1240p)")
    t0 = time.time()

    INT8_DIR.mkdir(parents=True, exist_ok=True)

    quantizer = ORTQuantizer.from_pretrained(str(FP32_DIR))

    # AVX512_VNNI is the right choice for i5-1240p (supports VNNI instructions)
    # Falls back gracefully at runtime if VNNI not available
    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=False,          # dynamic quant — no calibration dataset needed
        per_channel=False,        # per-tensor is faster on small batches
        operators_to_quantize=["MatMul", "Gemm"],
    )

    quantizer.quantize(
        save_dir=str(INT8_DIR),
        quantization_config=qconfig,
    )
    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.0f}s → {INT8_DIR}")

    # ── Step 4: Verify the INT8 model loads and runs ──────────────
    print("\n[3/3] Smoke-testing INT8 model...")
    from transformers import AutoTokenizer
    import numpy as np

    tok   = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = ORTModelForSequenceClassification.from_pretrained(
        str(INT8_DIR),
        provider="CPUExecutionProvider",
    )

    # One inference pass
    t0 = time.time()
    enc = tok(["test query"], ["test passage"], return_tensors="pt",
              truncation=True, max_length=512)
    out = model(**enc)
    ms = (time.time() - t0) * 1000
    print(f"      Single pair inference: {ms:.0f}ms — model is working")

    # Size comparison
    fp32_size = sum(f.stat().st_size for f in FP32_DIR.rglob("*") if f.is_file())
    int8_size = sum(f.stat().st_size for f in INT8_DIR.rglob("*") if f.is_file())
    print(f"      fp32 size: {fp32_size/1e9:.2f} GB")
    print(f"      INT8 size: {int8_size/1e9:.2f} GB  ({100*int8_size/fp32_size:.0f}% of fp32)")

    print("\n" + "=" * 60)
    print("  SUCCESS — add to your .env:")
    print(f"  RERANKER_INT8_PATH={INT8_DIR.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()