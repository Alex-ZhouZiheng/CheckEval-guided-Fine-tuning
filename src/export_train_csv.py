# export_tb_csv.py
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def safe_name(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_scalars_from_path(path: Path):
    ea = event_accumulator.EventAccumulator(
        str(path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        data[tag] = ea.Scalars(tag)
    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_tb_csv.py <event_file_or_log_dir> [output_dir]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Path not found: {input_path}")
        sys.exit(2)

    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path / "csv_export" if input_path.is_dir() else input_path.parent / "csv_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        event_files = sorted(input_path.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"No event files found in: {input_path}")
            sys.exit(3)
    else:
        event_files = [input_path]

    merged = defaultdict(list)

    for event_file in event_files:
        print(f"Loading: {event_file}")
        scalars = load_scalars_from_path(event_file)
        for tag, events in scalars.items():
            for e in events:
                merged[tag].append(
                    {
                        "step": e.step,
                        "value": e.value,
                        "wall_time": e.wall_time,
                        "source_file": event_file.name,
                    }
                )

    if not merged:
        print("No scalar tags found.")
        sys.exit(4)

    # 每个 tag 单独一个 csv
    for tag, rows in merged.items():
        rows = sorted(rows, key=lambda x: (x["step"], x["wall_time"]))
        out_file = out_dir / f"{safe_name(tag)}.csv"
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "value", "wall_time", "source_file"])
            writer.writeheader()
            writer.writerows(rows)

    # long format
    long_rows = []
    for tag, rows in merged.items():
        for r in rows:
            long_rows.append(
                {
                    "tag": tag,
                    "step": r["step"],
                    "value": r["value"],
                    "wall_time": r["wall_time"],
                    "source_file": r["source_file"],
                }
            )
    long_rows.sort(key=lambda x: (x["tag"], x["step"], x["wall_time"]))

    long_file = out_dir / "all_scalars_long.csv"
    with long_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "step", "value", "wall_time", "source_file"])
        writer.writeheader()
        writer.writerows(long_rows)

    # wide format: 同一步如果重复，保留 wall_time 最晚的一条
    step_map = defaultdict(dict)
    for tag, rows in merged.items():
        by_step = {}
        for r in rows:
            step = r["step"]
            if step not in by_step or r["wall_time"] > by_step[step]["wall_time"]:
                by_step[step] = r
        for step, r in by_step.items():
            step_map[step][tag] = r["value"]

    wide_file = out_dir / "all_scalars_wide.csv"
    scalar_tags = sorted(merged.keys())
    with wide_file.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["step"] + scalar_tags
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(step_map):
            row = {"step": step}
            row.update(step_map[step])
            writer.writerow(row)

    print(f"\nExported {len(scalar_tags)} scalar tags from {len(event_files)} event files")
    print(f"Output dir: {out_dir}")
    print(f"Main file: {wide_file}")


if __name__ == "__main__":
    main()
