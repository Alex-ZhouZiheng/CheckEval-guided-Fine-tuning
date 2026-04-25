#!/usr/bin/env python3
"""Convert HF checkpoints and PEFT LoRA adapters to llama.cpp GGUF.

Wraps three scripts from a local llama.cpp clone (pointed to by
``$LLAMA_CPP_HOME``):
- ``convert_hf_to_gguf.py``  : HF model dir -> base f16 GGUF
- ``llama-quantize``         : f16 GGUF -> quantized (Q4_K_M, Q5_K_M, ...)
- ``convert_lora_to_gguf.py``: PEFT adapter dir -> GGUF-LoRA

Examples:
    # Base model
    python src/train/convert_to_gguf.py base \
        --hf-path models/Qwen3.5-9B \
        --out   models/gguf/Qwen3.5-9B \
        --quant Q4_K_M

    # LoRA adapter
    python src/train/convert_to_gguf.py lora \
        --adapter-path results/checkpoints/run1/final_adapter \
        --base         models/Qwen3.5-9B \
        --out          models/gguf/adapters/run1.gguf
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _llama_cpp_home() -> Path:
    home = os.environ.get("LLAMA_CPP_HOME")
    if not home:
        raise SystemExit(
            "Set $LLAMA_CPP_HOME to the root of a built llama.cpp clone "
            "(containing convert_hf_to_gguf.py and build/bin/llama-quantize)."
        )
    p = Path(home).expanduser()
    if not p.exists():
        raise SystemExit(f"$LLAMA_CPP_HOME={p} does not exist.")
    return p


def _resolve_tool(home: Path, *candidates: str) -> Path:
    for c in candidates:
        candidate = home / c
        if candidate.exists():
            return candidate
    which = shutil.which(candidates[-1])
    if which:
        return Path(which)
    raise SystemExit(f"Could not find any of {candidates} under {home} or $PATH")


def _run(cmd: list[str]) -> None:
    log.info("+ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed (exit {result.returncode}): {cmd}")


def convert_base(hf_path: Path, out_stem: Path, quant: str) -> Path:
    home = _llama_cpp_home()
    convert_hf = _resolve_tool(home, "convert_hf_to_gguf.py")
    quantize_bin = _resolve_tool(
        home,
        "build/bin/llama-quantize",
        "build/bin/Release/llama-quantize.exe",
        "llama-quantize",
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    f16_path = out_stem.with_suffix(".f16.gguf")
    quant_path = out_stem.with_suffix(f".{quant}.gguf")

    if f16_path.exists():
        log.info("Reusing existing f16 GGUF: %s", f16_path)
    else:
        _run([
            sys.executable, str(convert_hf),
            str(hf_path),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ])

    _run([str(quantize_bin), str(f16_path), str(quant_path), quant])
    log.info("Wrote %s", quant_path)
    return quant_path


def convert_lora(adapter_path: Path, base_hf: Path, out_path: Path) -> Path:
    home = _llama_cpp_home()
    convert_lora = _resolve_tool(home, "convert_lora_to_gguf.py")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, str(convert_lora),
        str(adapter_path),
        "--base", str(base_hf),
        "--outfile", str(out_path),
    ])
    log.info("Wrote %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("base", help="Convert + quantize an HF model directory.")
    p_base.add_argument("--hf-path", type=Path, required=True)
    p_base.add_argument("--out", type=Path, required=True,
                        help="Output stem; quantized file will be <out>.<QUANT>.gguf")
    p_base.add_argument("--quant", type=str, default="Q4_K_M")

    p_lora = sub.add_parser("lora", help="Convert a PEFT LoRA adapter to GGUF-LoRA.")
    p_lora.add_argument("--adapter-path", type=Path, required=True)
    p_lora.add_argument("--base", type=Path, required=True,
                        help="Base HF model dir used to train the adapter.")
    p_lora.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "base":
        convert_base(args.hf_path, args.out, args.quant)
    elif args.cmd == "lora":
        convert_lora(args.adapter_path, args.base, args.out)


if __name__ == "__main__":
    main()
