#!/usr/bin/env python3
"""
Control experiments for checklist source and quality.

Prepares SFT data under different checklist configurations, then trains
joint models for comparison.

Experiment axes:
  1. Checklist SOURCE:  filtered (default) vs v2
  2. Checklist QUALITY: full vs ablated (drop one dimension at a time)
  3. Lambda SWEEP:      λ ∈ {0.0, 0.05, 0.1, 0.2, 0.5}

Usage:
    # Run all control experiments on debug_5k (quick test)
    python run_checklist_control.py --tier debug_5k --no-wandb

    # Run only the source experiment on tier_10k
    python run_checklist_control.py --tier tier_10k --experiment source

    # Run only the lambda sweep
    python run_checklist_control.py --tier tier_10k --experiment lambda

    # Run ablation experiments (drop each dimension)
    python run_checklist_control.py --tier tier_10k --experiment ablation

    # Dry run: just prepare SFT data, don't train
    python run_checklist_control.py --tier debug_5k --prepare-only
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = cfg.PROJECT_ROOT
CHECKLISTS_BASE = PROJECT_ROOT / "checklists"
CONTROL_DIR = cfg.DATA_DIR / "checklist_sft_control"
CONTROL_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────── checklist manipulation ─────────────


def create_ablated_checklist(
    source_dir: Path,
    drop_dimension: str,
    output_dir: Path,
) -> None:
    """Copy checklists but exclude one dimension entirely."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for yaml_path in sorted(source_dir.glob("*_filtered.yaml")):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        dim_name = data.get("dimension", yaml_path.stem)
        if dim_name == drop_dimension:
            log.info("  Ablation: dropping %s", dim_name)
            continue

        out_path = output_dir / yaml_path.name
        shutil.copy2(yaml_path, out_path)

    log.info("  Ablated checklist → %s", output_dir)


def create_random_subset_checklist(
    source_dir: Path,
    keep_fraction: float,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Keep a random fraction of questions from each dimension."""
    import random
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for yaml_path in sorted(source_dir.glob("*_filtered.yaml")):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        for sub_name, sub_data in data.get("sub_aspects", {}).items():
            questions = sub_data.get("filtered_questions", [])
            n_keep = max(1, int(len(questions) * keep_fraction))
            sub_data["filtered_questions"] = rng.sample(questions, n_keep)

        out_path = output_dir / yaml_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    log.info("  Random subset (%.0f%%) → %s", keep_fraction * 100, output_dir)


# ────────────────────────── experiment runners ─────────────────


def run_prepare(
    tier: str,
    checklist_dir: Path,
    output_dir: Path,
    mode: str = "synthetic",
    label: str = "",
) -> Path:
    """Run prepare_checklist_sft.py and return the output path."""
    tag = f"_{mode}" if mode == "synthetic" else ""
    out_name = f"train_{tier}{tag}.parquet"
    out_path = output_dir / out_name

    if out_path.exists():
        log.info("  [%s] SFT data already exists: %s", label, out_path)
        return out_path

    cmd = [
        sys.executable, "prepare_checklist_sft.py",
        "--tier", tier,
        "--mode", mode,
        "--checklist-dir", str(checklist_dir),
        "--output-dir", str(output_dir),
    ]
    log.info("  [%s] Preparing SFT data: %s", label, " ".join(cmd[-6:]))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT / "src"))
    return out_path


def run_train(
    tier: str,
    sft_data: Path,
    sft_lambda: float,
    run_name: str,
    extra_args: list[str] | None = None,
) -> None:
    """Run run_joint_train.py."""
    cmd = [
        sys.executable, "run_joint_train.py",
        "--tier", tier,
        "--sft-data", str(sft_data),
        "--sft-lambda", str(sft_lambda),
        "--run-name", run_name,
    ]
    if extra_args:
        cmd.extend(extra_args)
    log.info("  Training: %s", run_name)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT / "src"))


def get_dimensions(checklist_dir: Path) -> list[str]:
    """List dimension names from YAML files."""
    dims = []
    for yaml_path in sorted(checklist_dir.glob("*_filtered.yaml")):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        dims.append(data.get("dimension", yaml_path.stem))
    return dims


# ────────────────────────── experiments ─────────────────────────


def experiment_source(tier: str, mode: str, extra_args: list[str],
                       prepare_only: bool) -> None:
    """Compare checklist sources: filtered vs v2."""
    log.info("=" * 50)
    log.info("Experiment: Checklist SOURCE")
    log.info("=" * 50)

    configs = [
        ("filtered", CHECKLISTS_BASE / "filtered"),
        ("v2",       CHECKLISTS_BASE / "v2"),
    ]

    for label, cl_dir in configs:
        if not cl_dir.exists():
            log.warning("  Checklist dir not found, skipping: %s", cl_dir)
            continue

        out_dir = CONTROL_DIR / f"source_{label}"
        sft_path = run_prepare(tier, cl_dir, out_dir, mode=mode, label=label)

        if not prepare_only:
            run_train(
                tier, sft_path, sft_lambda=cfg.JOINT_LAMBDA,
                run_name=f"joint_{tier}_src_{label}",
                extra_args=extra_args,
            )


def experiment_ablation(tier: str, mode: str, extra_args: list[str],
                          prepare_only: bool) -> None:
    """Drop one dimension at a time from checklists."""
    log.info("=" * 50)
    log.info("Experiment: Checklist QUALITY (dimension ablation)")
    log.info("=" * 50)

    source_dir = CHECKLISTS_BASE / "filtered"
    dims = get_dimensions(source_dir)
    log.info("  Dimensions: %s", dims)

    for drop_dim in dims:
        label = f"drop_{drop_dim}"
        ablated_dir = CONTROL_DIR / "checklists" / label
        create_ablated_checklist(source_dir, drop_dim, ablated_dir)

        out_dir = CONTROL_DIR / f"ablation_{label}"
        sft_path = run_prepare(tier, ablated_dir, out_dir, mode=mode, label=label)

        if not prepare_only:
            run_train(
                tier, sft_path, sft_lambda=cfg.JOINT_LAMBDA,
                run_name=f"joint_{tier}_abl_{drop_dim[:20]}",
                extra_args=extra_args,
            )

    # Also: random 50% subset
    label = "random_50pct"
    subset_dir = CONTROL_DIR / "checklists" / label
    create_random_subset_checklist(source_dir, 0.5, subset_dir)

    out_dir = CONTROL_DIR / f"quality_{label}"
    sft_path = run_prepare(tier, subset_dir, out_dir, mode=mode, label=label)

    if not prepare_only:
        run_train(
            tier, sft_path, sft_lambda=cfg.JOINT_LAMBDA,
            run_name=f"joint_{tier}_q_random50",
            extra_args=extra_args,
        )


def experiment_lambda(tier: str, mode: str, extra_args: list[str],
                        prepare_only: bool) -> None:
    """Sweep λ values: {0.0, 0.05, 0.1, 0.2, 0.5}."""
    log.info("=" * 50)
    log.info("Experiment: Lambda SWEEP")
    log.info("=" * 50)

    # Prepare SFT data once (using default checklists)
    out_dir = CONTROL_DIR / "source_filtered"
    sft_path = run_prepare(
        tier, CHECKLISTS_BASE / "filtered", out_dir, mode=mode, label="lambda",
    )

    if prepare_only:
        return

    for lam in [0.0, 0.05, 0.1, 0.2, 0.5]:
        lam_str = f"{lam:.2f}".replace(".", "")
        run_train(
            tier, sft_path, sft_lambda=lam,
            run_name=f"joint_{tier}_lam{lam_str}",
            extra_args=extra_args,
        )


# ────────────────────────── main ───────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run checklist control experiments"
    )
    parser.add_argument("--tier", type=str, default="debug_5k",
                        choices=["debug_5k", "tier_10k", "tier_20k", "full"])
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "source", "ablation", "lambda"],
                        help="Which experiment to run")
    parser.add_argument("--mode", type=str, default="synthetic",
                        choices=["synthetic", "teacher"],
                        help="SFT data generation mode")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare SFT data, skip training")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    extra_args = []
    if args.no_wandb:
        extra_args.append("--no-wandb")
    if args.no_tensorboard:
        extra_args.append("--no-tensorboard")

    experiments = {
        "source":   experiment_source,
        "ablation": experiment_ablation,
        "lambda":   experiment_lambda,
    }

    if args.experiment == "all":
        for name, fn in experiments.items():
            fn(args.tier, args.mode, extra_args, args.prepare_only)
    else:
        experiments[args.experiment](
            args.tier, args.mode, extra_args, args.prepare_only,
        )

    log.info("All experiments done.")


if __name__ == "__main__":
    main()
