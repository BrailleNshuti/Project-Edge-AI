"""Formats the app's Evaluate-mode results.csv into report-ready markdown tables.

The Android app's Evaluate mode already computes and displays subgroup accuracy,
but this turns the raw per-image results.csv it writes into clean tables for
copy-pasting into the actual report appendix.

Usage:
  python evaluate_20_images.py /path/to/results.csv
"""
import argparse
import csv
import pathlib
from collections import defaultdict


def _accuracy_table(rows: list[dict], gt_col: str, pred_col: str, label: str) -> str:
    by_group = defaultdict(lambda: [0, 0])  # group -> [correct, total]
    for row in rows:
        group = row[gt_col]
        by_group[group][1] += 1
        if row[gt_col].strip().lower() == row[pred_col].strip().lower():
            by_group[group][0] += 1

    total_correct = sum(c for c, _ in by_group.values())
    total_n = sum(n for _, n in by_group.values())

    lines = [f"**{label} accuracy**", "", "| Group | Accuracy | N |", "|---|---|---|"]
    for group in sorted(by_group):
        correct, n = by_group[group]
        lines.append(f"| {group} | {correct / n * 100:.1f}% | {n} |")
    lines.append(f"| **Overall** | **{total_correct / total_n * 100:.1f}%** | **{total_n}** |")
    return "\n".join(lines)


def main(csv_path: pathlib.Path) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"{csv_path} has no rows.")

    sections = [
        _accuracy_table(rows, "gt_age_group", "pred_age_group", "Age group"),
        _accuracy_table(rows, "gt_gender", "pred_gender", "Gender"),
        _accuracy_table(rows, "gt_expression", "pred_expression", "Expression"),
    ]
    output = "\n\n".join(sections) + "\n"

    out_path = csv_path.parent / "evaluation_tables.md"
    out_path.write_text(output)
    print(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=pathlib.Path, help="results.csv exported by the app's Evaluate mode")
    args = parser.parse_args()
    main(args.results_csv)
