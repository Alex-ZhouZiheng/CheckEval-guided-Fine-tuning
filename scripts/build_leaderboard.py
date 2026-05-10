"""Build canonical leaderboard from normalized metric records.

Consumes `results/normalized/*.json`. Filters to a single split (default
`test`). Emits `main_table.{md,csv,json}` to `--out-dir`. Refuses to mix
splits or n_total values.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORM_DIR = ROOT / "results" / "normalized"


def load_norm_records(norm_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(norm_dir.glob("*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm-dir", default=str(NORM_DIR))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include", nargs="*", help="filter normalized basenames (substring match); else all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_norm_records(Path(args.norm_dir))
    records = [r for r in records if r["split"]["name"] == args.split]
    if args.include:
        records = [r for r in records if any(s in r["source_file"] for s in args.include)]

    if not records:
        print("no records match")
        return 1

    n_totals = {r["counts"]["n_total"] for r in records}
    if len(n_totals) > 1:
        print(f"n_total mismatch across rows: {n_totals}")
        return 2
    split_shas = {r["split"]["split_sha256"] for r in records}
    if len(split_shas) > 1:
        print(f"split_sha mismatch across rows: {split_shas}")
        return 3

    n_total = next(iter(n_totals))
    split_sha = next(iter(split_shas))

    rows = []
    for r in records:
        m = r["metrics"]
        c = r["counts"]
        rows.append({
            "method": r["method"],
            "family": r["family"],
            "real_acc": m["real_accuracy"],
            "valid_acc": m["valid_accuracy"],
            "parse_rate": m["parse_rate"],
            "n_valid": c["n_valid"],
            "n_total": c["n_total"],
            "macro_f1": m["macro_f1"],
            "position_bias_A": m["position_bias_A"],
            "base_id": r["model"]["base_id"],
            "adapter_path": r["model"]["adapter_path"],
            "bank": r["checklist"]["bank"],
            "split_sha256": r["split"]["split_sha256"],
            "source_file": r["source_file"],
        })
    rows.sort(key=lambda r: r["real_acc"] or 0.0, reverse=True)

    csv_path = out_dir / "main_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "main_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Canonical leaderboard — split=`{args.split}` n_total={n_total} "
                f"split_sha256=`{(split_sha or 'NA')[:16]}`\n\n")
        f.write("| # | Method | Family | Real acc | Valid acc | Parse | n_valid | macro_f1 | bank |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for i, r in enumerate(rows, 1):
            f.write(f"| {i} | {r['method']} | {r['family']} | "
                    f"{(r['real_acc'] or 0):.4f} | {(r['valid_acc'] or 0):.4f} | "
                    f"{(r['parse_rate'] or 0):.4f} | {r['n_valid']} | "
                    f"{(r['macro_f1'] or 0):.4f} | {r['bank'] or '—'} |\n")

    json_path = out_dir / "main_table.json"
    json_path.write_text(json.dumps({
        "split": args.split,
        "n_total": n_total,
        "split_sha256": split_sha,
        "rows": rows,
    }, indent=2), encoding="utf-8")

    print(f"wrote {csv_path}\nwrote {md_path}\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
