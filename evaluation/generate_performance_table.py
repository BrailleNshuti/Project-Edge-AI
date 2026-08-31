"""Builds the report's CPU-vs-GPU-vs-edge performance table from:
  - training/results/age_gender_cpu.json   (written by train_age_gender.py --tag cpu)
  - training/results/age_gender_gpu.json   (written by the same script on Colab, --tag gpu)
  - training/results/expression_cpu.json
  - training/results/expression_gpu.json
  - evaluation/edge_results.json           (you fill this in from the app's Benchmark screen)

Usage:
  python generate_performance_table.py
Writes evaluation/performance_table.md (paste directly into the report).
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "training" / "results"
EDGE_RESULTS_PATH = ROOT / "evaluation" / "edge_results.json"
OUT_PATH = ROOT / "evaluation" / "performance_table.md"


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_edge() -> dict:
    if not EDGE_RESULTS_PATH.exists():
        raise SystemExit(
            f"{EDGE_RESULTS_PATH} not found. Copy evaluation/EDGE_RESULTS_TEMPLATE.json there "
            "and fill in the numbers shown on the app's Benchmark screen."
        )
    with open(EDGE_RESULTS_PATH) as f:
        return json.load(f)


def _fmt(value, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.2f}{suffix}"


def _pct(value):
    """Keras metrics report accuracy as a 0-1 fraction; scale to 0-100 to match
    the percentages evaluate_20_images.py already writes for the edge column."""
    return None if value is None else value * 100


def main() -> None:
    ag_cpu = _load("age_gender_cpu")
    ag_gpu = _load("age_gender_gpu")
    ex_cpu = _load("expression_cpu")
    ex_gpu = _load("expression_gpu")
    edge = _load_edge()

    def get(d: dict | None, *keys):
        if d is None:
            return None
        v = d
        for k in keys:
            if v is None:
                return None
            v = v.get(k)
        return v

    lines = [
        "| Metric | Trained on CPU | Trained on GPU | On edge device |",
        "|---|---|---|---|",
        f"| Age MAE (years, continuous) | {_fmt(get(ag_cpu, 'test_metrics', 'age_mae'))} "
        f"| {_fmt(get(ag_gpu, 'test_metrics', 'age_mae'))} | -- |",
        f"| Age bucket accuracy (adult/elderly) | {_fmt(_pct(get(ag_cpu, 'test_metrics', 'age_bucket_accuracy')), '%')} "
        f"| {_fmt(_pct(get(ag_gpu, 'test_metrics', 'age_bucket_accuracy')), '%')} "
        f"| {_fmt(edge.get('age_bucket_accuracy_on_20_images'), '%')} |",
        f"| Gender accuracy | {_fmt(_pct(get(ag_cpu, 'test_metrics', 'gender_accuracy')), '%')} "
        f"| {_fmt(_pct(get(ag_gpu, 'test_metrics', 'gender_accuracy')), '%')} | {_fmt(edge.get('gender_accuracy_on_20_images'), '%')} |",
        f"| Expression accuracy | {_fmt(_pct(get(ex_cpu, 'test_metrics', 'accuracy')), '%')} "
        f"| {_fmt(_pct(get(ex_gpu, 'test_metrics', 'accuracy')), '%')} | {_fmt(edge.get('expression_accuracy_on_20_images'), '%')} |",
        f"| Training time (seconds, age/gender model) | {_fmt(get(ag_cpu, 'train_seconds'))} "
        f"| {_fmt(get(ag_gpu, 'train_seconds'))} | -- |",
        f"| Training time (seconds, expression model) | {_fmt(get(ex_cpu, 'train_seconds'))} "
        f"| {_fmt(get(ex_gpu, 'train_seconds'))} | -- |",
        f"| Face detection latency (ms) | -- | -- | {_fmt(edge.get('face_detect_latency_ms'))} |",
        f"| Age/gender inference latency (ms) | -- | -- | {_fmt(edge.get('age_gender_latency_ms'))} |",
        f"| Expression inference latency (ms) | -- | -- | {_fmt(edge.get('expression_latency_ms'))} |",
    ]
    table = "\n".join(lines)

    OUT_PATH.write_text(table + "\n")
    print(table)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
