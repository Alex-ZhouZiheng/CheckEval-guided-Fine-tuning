#!/usr/bin/env python3
"""Freeze a checklist bank and build a flat global question index.

Usage:
    python src/data_process/build_bank_index.py --bank checklists/v3 --out checklists/v3_frozen
"""

from __future__ import annotations

import argparse
import json
import logging
import os as _os
import shutil
import sys as _sys
from pathlib import Path

import pandas as pd
import yaml

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from utils import load_checklists, _select_dimensions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _iter_bank_rows(bank_dir: Path) -> list[dict]:
    rows: list[dict] = []
    qid = 1

    for yaml_path in sorted(bank_dir.glob("*_filtered.yaml")):
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        dimension = data.get("dimension", yaml_path.stem)
        definition = data.get("definition", "")

        for sub_aspect, sub_data in (data.get("sub_aspects") or {}).items():
            # Match existing load_checklists() behavior: dedupe only inside each sub-aspect.
            seen: set[str] = set()
            for question in sub_data.get("filtered_questions", []) or []:
                q_text = str(question).strip()
                if not q_text or q_text in seen:
                    continue
                seen.add(q_text)
                rows.append(
                    {
                        "qid": qid,
                        "dimension": dimension,
                        "sub_aspect": str(sub_aspect),
                        "question_text": q_text,
                        "definition": definition,
                        "source_yaml": yaml_path.name,
                    }
                )
                qid += 1

    return rows


def _copy_bank_files(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    if dst_dir.exists() and any(dst_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{dst_dir} already exists and is not empty. Use --overwrite to refresh."
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.iterdir()):
        target = dst_dir / path.name
        if path.is_dir():
            if target.exists() and overwrite:
                shutil.rmtree(target)
            shutil.copytree(path, target, dirs_exist_ok=overwrite)
        else:
            shutil.copy2(path, target)


def _summarize(df: pd.DataFrame) -> None:
    log.info("Indexed %d questions across %d dimensions", len(df), df["dimension"].nunique())
    for dim, g in df.groupby("dimension", sort=False):
        log.info("  %s: %d", dim, len(g))


def _assert_matches_load_checklists(df: pd.DataFrame, bank_dir: Path) -> None:
    """Cross-check flat qid order against utils.load_checklists().

    load_checklists() is the source of truth used by run_checkeval_judge.py
    (which produced the 77.05% v3 baseline).  Any drift between the flat
    bank_index produced here and load_checklists() means every oracle label
    will be mis-indexed.
    """
    checklists, _ = load_checklists(bank_dir)

    ref_flat: list[tuple[str, str]] = []
    for dim_name, questions in checklists.items():
        for q in questions:
            ref_flat.append((dim_name, q))

    our_flat = [(str(r["dimension"]), str(r["question_text"])) for _, r in df.iterrows()]

    if len(ref_flat) != len(our_flat):
        raise AssertionError(
            f"qid alignment mismatch: load_checklists()={len(ref_flat)} "
            f"but bank_index={len(our_flat)}"
        )

    for i, (ref, ours) in enumerate(zip(ref_flat, our_flat), start=1):
        if ref != ours:
            raise AssertionError(
                f"qid {i} drift: load_checklists()={ref!r} vs bank_index={ours!r}"
            )

    log.info("qid alignment OK: %d questions match load_checklists() order", len(df))


def _assert_matches_baseline_metrics(df: pd.DataFrame, metrics_path: Path) -> None:
    """Optional cross-check against existing v3 baseline metrics JSON.

    Verifies per-domain effective question counts (shared_dims + optional code)
    are consistent with n_questions seen in the baseline run.
    """
    if not metrics_path.exists():
        log.info("Baseline metrics %s not found — skipping cross-check", metrics_path)
        return

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    per_domain = metrics.get("per_domain") or {}
    if not per_domain:
        log.info("Baseline metrics has no per_domain block — skipping cross-check")
        return

    dim_to_count = df.groupby("dimension").size().to_dict()

    for domain in ("general", "stem", "code"):
        allowed = _select_dimensions(domain)
        expected = sum(
            int(dim_to_count.get(d, 0))
            for d in dim_to_count
            if d in allowed or str(d).lower() in {a.lower() for a in allowed}
        )
        observed = per_domain.get(domain, {}).get("n_questions")
        if observed is None:
            continue
        if int(observed) != int(expected):
            log.warning(
                "Domain %s: bank_index implies n_questions=%d but baseline reports %d",
                domain,
                expected,
                observed,
            )
        else:
            log.info("Domain %s per-question count %d matches baseline", domain, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True, help="Source bank dir, e.g. checklists/v3")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Frozen output dir. Writes bank_index.parquet under this directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("results/checkeval_pairwise_naaware_dev_600_v3_q9b_metrics.json"),
        help="Optional baseline metrics JSON for per-domain sanity check",
    )
    parser.add_argument(
        "--skip-alignment-check",
        action="store_true",
        help="Skip cross-check against utils.load_checklists() (DANGEROUS)",
    )
    args = parser.parse_args()

    bank_dir = args.bank.resolve()
    out_dir = args.out.resolve()
    if not bank_dir.exists():
        raise FileNotFoundError(bank_dir)

    rows = _iter_bank_rows(bank_dir)
    if not rows:
        raise SystemExit(f"No *_filtered.yaml files found under {bank_dir}")

    df = pd.DataFrame(rows).sort_values("qid", kind="stable").reset_index(drop=True)
    _summarize(df)

    if not args.skip_alignment_check:
        _assert_matches_load_checklists(df, bank_dir)
    else:
        log.warning("Skipping alignment check against load_checklists() — qids may drift")

    _assert_matches_baseline_metrics(df, args.baseline_metrics)

    _copy_bank_files(bank_dir, out_dir, overwrite=args.overwrite)

    out_path = out_dir / "bank_index.parquet"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite to replace it.")

    df.to_parquet(out_path, index=False)
    log.info("Saved bank index -> %s", out_path)


if __name__ == "__main__":
    main()
