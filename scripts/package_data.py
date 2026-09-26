"""Create a portable prepared-data bundle from the exact ordered release splits."""

import argparse
import tarfile
from pathlib import Path
from check_data import ROOT, TAGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(TAGS), default=["coach420", "holo4k"]
    )
    parser.add_argument(
        "--partition", choices=["train", "valid", "test", "all"], default="test"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("Duplicate datasets")
    selected = []
    for dataset in args.datasets:
        name = f"{args.partition}_ids_{TAGS[dataset]}"
        split = ROOT / "data/splits" / dataset / name
        ids = split.read_text().split()
        if (args.data_root / dataset / "splits" / name).read_text().split() != ids:
            raise ValueError(f"Split mismatch: {dataset}/{name}")
        selected.append((split, Path(dataset) / "splits" / name))
        for sample in ids:
            relative = Path(dataset) / "protein_ligand" / sample
            for filename in [
                "protein.pdb",
                "source_info.csv",
                "source.mapping",
                "site_manifest.csv",
            ]:
                path = args.data_root / relative / filename
                if filename == "protein.pdb" or path.is_file():
                    selected.append((path, relative / filename))
            for pattern in ["ligand*", "pocket*.txt"]:
                for path in sorted((args.data_root / relative).glob(pattern)):
                    if path.is_file():
                        selected.append((path, relative / path.name))
            for modal, suffix in [("esm", ".npy"), ("pocket", ".npz")]:
                relative = Path(dataset) / modal / f"{sample}{suffix}"
                selected.append((args.data_root / relative, relative))
    for source, _ in selected:
        if not source.is_file():
            raise FileNotFoundError(source)
    if len({str(name) for _, name in selected}) != len(selected):
        raise ValueError("Duplicate archive entries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Private sample manifests, original paths and unrelated modalities are excluded.
    with tarfile.open(args.output, "x") as archive:
        for source, name in selected:
            info = archive.gettarinfo(str(source), arcname=str(name))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as handle:
                archive.addfile(info, handle)
    print(f"Wrote {len(selected)} files to {args.output}")


if __name__ == "__main__":
    main()
