"""Generate side-by-side comparison of all 8 conditions.

Reads metrics.json from each condition dir and from existing pointwise results.
Produces comparison_summary.md.

Usage:
    python scripts/generate_comparison_summary.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CONDITIONS = {
    "C1_gen_pointwise_weighted": {
        "path": "results/checkeval_pairwise_naaware_dev_600_v3_q9b_metrics.json",
    },
    "C2_gen_comparative_count": {
        "path": "results/comparative/gen_comparative_count/metrics.json",
    },
    "C3_gen_comparative_weighted": {
        "path": "results/comparative/gen_comparative_weighted/metrics.json",
    },
    "C4_hr_pointwise_weighted": {
        "path": "results/checkeval_pairwise_naaware_dev_600_hr_q9b_metrics.json",
    },
    "C5_hr_comparative_count": {
        "path": "results/comparative/hr_comparative_count/metrics.json",
    },
    "C6_hr_comparative_weighted": {
        "path": "results/comparative/hr_comparative_weighted/metrics.json",
    },
    "C7_fullbank_pointwise": {
        "path": "results/checkeval_pairwise_naaware_dev_600_v4_q9b_metrics.json",
    },
    "C8_fullbank_comparative_count": {
        "path": "results/comparative/fullbank_comparative_count/metrics.json",
    },
}


def load_metrics(name, info):
    path = BASE_DIR / info["path"]
    if not path.exists():
        print(f"  [warn] {name}: metrics not found at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, default="—"):
    if val is None or val == default:
        return default
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main():
    lines = []
    lines.append("# Comparative Checklist Evaluation — Comparison Summary\n")
    lines.append("## Side-by-Side Results\n")
    hdr = "| Condition | Question Source | Format | Aggregation | Accuracy | Tie Rate | Eff. Accuracy | Parse Rate | Valid/Total |"
    sep = "|-----------|----------------|--------|-------------|----------|----------|---------------|------------|-------------|"
    lines.append(hdr)
    lines.append(sep)

    for name in ["C1_gen_pointwise_weighted", "C2_gen_comparative_count",
                  "C3_gen_comparative_weighted", "C4_hr_pointwise_weighted",
                  "C5_hr_comparative_count", "C6_hr_comparative_weighted",
                  "C7_fullbank_pointwise", "C8_fullbank_comparative_count"]:
        info = CONDITIONS[name]
        m = load_metrics(name, info)
        if m is None:
            lines.append(f"| {name} | — | — | — | — | — | — | — | — |")
        else:
            src = name.split("_")[1]  # gen, hr, fullbank
            fmt_ = "comparative" if "comparative" in name else "pointwise"
            agg = name.split("_")[-1] if "comparative" in name else "weighted"
            lines.append(
                f"| {name} | {src} | {fmt_} | {agg} "
                f"| {fmt(m.get('accuracy'))} "
                f"| {fmt(m.get('tie_rate'))} "
                f"| {fmt(m.get('effective_accuracy'))} "
                f"| {fmt(m.get('parse_ok_rate'))} "
                f"| {fmt(m.get('n_valid'))}/{fmt(m.get('n_total'))} |"
            )

    lines.append("")
    lines.append("## Per-Question-Source Delta\n")
    lines.append("| Source | Pointwise Acc | Comparative Acc | Δ Accuracy | Pointwise Tie Rate | Comparative Tie Rate | Δ Tie Rate |")
    lines.append("|--------|---------------|-----------------|------------|-------------------|---------------------|------------|")

    deltas = {
        "generated": ("C1_gen_pointwise_weighted", ["C2_gen_comparative_count", "C3_gen_comparative_weighted"]),
        "hr-oracle": ("C4_hr_pointwise_weighted", ["C5_hr_comparative_count", "C6_hr_comparative_weighted"]),
        "fullbank": ("C7_fullbank_pointwise", ["C8_fullbank_comparative_count"]),
    }
    for src_name, (pw_name, comp_names) in deltas.items():
        pw_m = load_metrics(pw_name, CONDITIONS[pw_name])
        best_comp_m = None
        for cn in comp_names:
            cm = load_metrics(cn, CONDITIONS[cn])
            if cm and (best_comp_m is None):
                best_comp_m = cm
            elif cm and best_comp_m:
                # Pick better effective accuracy
                ca = cm.get("effective_accuracy", 0) or 0
                ba = best_comp_m.get("effective_accuracy", 0) or 0
                if ca > ba:
                    best_comp_m = cm

        if pw_m and best_comp_m:
            pw_acc = pw_m.get("effective_accuracy", 0) or 0
            comp_acc = best_comp_m.get("effective_accuracy", 0) or 0
            pw_tie = pw_m.get("tie_rate", 0) or 0
            comp_tie = best_comp_m.get("tie_rate", 0) or 0
            lines.append(
                f"| {src_name} | {pw_acc:.4f} | {comp_acc:.4f} | {comp_acc - pw_acc:+.4f} "
                f"| {pw_tie:.4f} | {comp_tie:.4f} | {comp_tie - pw_tie:+.4f} |"
            )
        else:
            lines.append(f"| {src_name} | — | — | — | — | — | — |")

    lines.append("")
    lines.append("## A/B/Tie Answer Distribution\n")
    lines.append("| Condition | Avg A Count | Avg B Count | Avg Tie Count | Avg Answers/Sample |")
    lines.append("|-----------|-------------|-------------|---------------|--------------------|")
    for name in ["C2_gen_comparative_count", "C3_gen_comparative_weighted",
                  "C5_hr_comparative_count", "C6_hr_comparative_weighted",
                  "C8_fullbank_comparative_count"]:
        m = load_metrics(name, CONDITIONS[name])
        if m:
            lines.append(
                f"| {name} "
                f"| {fmt(m.get('avg_a_count'))} "
                f"| {fmt(m.get('avg_b_count'))} "
                f"| {fmt(m.get('avg_tie_count'))} "
                f"| {fmt(m.get('avg_answers_per_sample'))} |"
            )

    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("Comparative format is better if vs pointwise counterpart (same questions):")
    lines.append("- **Higher effective accuracy** — fewer yes/yes collapses")
    lines.append("- **Lower tie rate** — comparative forces decision per criterion")
    lines.append("- **Parse reliability >= 95%** — new format understood by judge")
    lines.append("")
    lines.append("Core hypothesis: simultaneous side-by-side comparison + forced A/B choice")
    lines.append("reduces the yes/yes collapse that dominates current pointwise errors.")

    out_path = BASE_DIR / "results" / "comparative" / "comparison_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison summary written to {out_path}")


if __name__ == "__main__":
    main()
