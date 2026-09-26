"""Check required files and exact split membership/order before evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = {"unisite": "unisite_0p9", "coach420": "coach420", "holo4k": "holo4k"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(TAGS), default=["coach420", "holo4k"]
    )
    parser.add_argument(
        "--partition", choices=["train", "valid", "test", "all"], default="test"
    )
    args = parser.parse_args()
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("Duplicate datasets")
    records = []
    for dataset in args.datasets:
        name = f"{args.partition}_ids_{TAGS[dataset]}"
        canonical = ROOT / "data/splits" / dataset / name
        expected = canonical.read_text().split()
        split = args.data_root / dataset / "splits" / name
        actual = split.read_text().split()
        if actual != expected:
            raise ValueError(f"Split order or membership mismatch: {split}")
        if len(expected) != len(set(expected)):
            raise ValueError("Repeated canonical sample ids")
        for sample in expected:
            if Path(sample).name != sample:
                raise ValueError(f"Invalid sample id {sample}")
            paths = [
                args.data_root / dataset / "protein_ligand" / sample / "protein.pdb",
                args.data_root / dataset / "pocket" / f"{sample}.npz",
                args.data_root / dataset / "esm" / f"{sample}.npy",
            ]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
        records.append(
            dict(
                dataset=dataset,
                partition=args.partition,
                proteins=len(expected),
                split_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest(),
            )
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
