"""Compare an evaluation CSV with the published main-checkpoint results."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "ap_0.3": "ap_iou_0.3",
    "ap_0.5": "ap_iou_0.5",
    "dcc_4": "query_dcc_topn",
    "dca_4": "query_dca_topn",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance on 0–1 scores; default 0.001 percentage point.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.atol < 1:
        raise ValueError("atol must be in [0, 1)")
    rows = list(csv.DictReader(args.metrics.open()))
    measured = {r["metric"]: float(r["value"]) for r in rows}
    if len(measured) != len(rows):
        raise ValueError("Duplicate metric rows")
    records = []
    for expected in csv.DictReader((ROOT / "results/expected.csv").open()):
        for key, metric in FIELDS.items():
            name = f"test/{expected['dataset']}/{metric}"
            value, reference = measured[name], float(expected[key])
            error = abs(value - reference)
            records.append(
                dict(
                    dataset=expected["dataset"],
                    metric=key,
                    expected=reference,
                    measured=value,
                    absolute_error=error,
                    passed=error <= args.atol,
                )
            )
    result = dict(
        atol=args.atol, passed=all(r["passed"] for r in records), metrics=records
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(
            "Main-result reproduction mismatch; inspect individual metrics"
        )


if __name__ == "__main__":
    main()
