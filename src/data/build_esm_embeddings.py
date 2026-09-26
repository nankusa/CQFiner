#!/usr/bin/env python3
"""Build chain-aware residue ESM embeddings into <dataset>/<output_modal>/<sample>.npy."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from click.core import ParameterSource
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel

from .config_utils import load_yaml_config, resolve_feature_build_config
from .modal_paths import modal_dir, modal_path


DEFAULT_HF_MODEL_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--facebook--esm2_t33_650M_UR50D"
)


@dataclass
class SampleRecord:
    sample_id: str
    sample_dir: Path
    output_path: Path
    chain_ids: list[str]
    chain_sequences: list[str]

    @property
    def residue_count(self) -> int:
        return sum(len(sequence) for sequence in self.chain_sequences)

    @property
    def chain_count(self) -> int:
        return len(self.chain_sequences)


@dataclass
class ChainTask:
    sample_id: str
    chain_index: int
    chain_id: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def _iter_sample_dirs(sample_root: Path) -> list[Path]:
    return sorted(
        [entry for entry in sample_root.iterdir() if entry.is_dir()],
        key=lambda path: path.name,
    )


def _extract_chain_sequences(pdb_path: Path) -> tuple[list[str], list[str]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    chain_ids: list[str] = []
    chain_sequences: list[str] = []
    for model in structure:
        for chain in model:
            residues: list[str] = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" not in residue:
                    continue
                try:
                    aa = seq1(residue.resname)
                except KeyError:
                    continue
                residues.append(aa)
            if residues:
                chain_ids.append(chain.id.strip() or "A")
                chain_sequences.append("".join(residues))
    return chain_ids, chain_sequences


def _read_sample(
    sample_dir: Path, dataset_root: Path, output_modal: str, overwrite: bool
) -> SampleRecord | None:
    sample_id = sample_dir.name
    protein_path = sample_dir / "protein.pdb"
    if not protein_path.is_file():
        raise FileNotFoundError(protein_path)

    output_path = modal_path(dataset_root, output_modal, sample_id, ".npy")
    if output_path.exists() and not overwrite:
        return None

    chain_ids, chain_sequences = _extract_chain_sequences(protein_path)
    if not chain_sequences:
        raise ValueError(f"No valid protein chains in {protein_path}")

    return SampleRecord(
        sample_id=sample_id,
        sample_dir=sample_dir,
        output_path=output_path,
        chain_ids=chain_ids,
        chain_sequences=chain_sequences,
    )


def _load_samples(
    sample_dirs: list[Path],
    dataset_root: Path,
    output_modal: str,
    overwrite: bool,
    n_jobs: int,
) -> list[SampleRecord]:
    if n_jobs <= 1:
        records = [
            _read_sample(
                sample_dir,
                dataset_root=dataset_root,
                output_modal=output_modal,
                overwrite=overwrite,
            )
            for sample_dir in tqdm(sample_dirs, desc="scan_sequences")
        ]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _read_sample,
                    sample_dir,
                    dataset_root,
                    output_modal,
                    overwrite,
                )
                for sample_dir in sample_dirs
            ]
            records = [
                future.result()
                for future in tqdm(futures, total=len(futures), desc="scan_sequences")
            ]
    return [record for record in records if record is not None]


def _iter_batches(tasks: list[ChainTask], batch_size: int, max_num_tokens: int | None):
    ordered = sorted(tasks, key=lambda item: item.length)
    batch: list[ChainTask] = []
    max_len = 0

    for item in ordered:
        sample_tokens = item.length + 2
        next_max_len = max(max_len, sample_tokens)
        next_batch_size = len(batch) + 1
        padded_tokens = next_max_len * next_batch_size
        over_tokens = max_num_tokens is not None and padded_tokens > max_num_tokens

        if batch and (next_batch_size > batch_size or over_tokens):
            yield batch
            batch = []
            max_len = 0

        batch.append(item)
        max_len = max(max_len, sample_tokens)

    if batch:
        yield batch


def _resolve_model_path(model_root: Path) -> Path:
    model_root = model_root.expanduser().resolve()
    if not (model_root / "config.json").is_file():
        raise FileNotFoundError(f"Pass the exact ESM model directory containing config.json: {model_root}")
    return model_root


def _write_embedding(output_path: Path, embedding: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    with tmp_path.open("wb") as handle:
        np.save(handle, embedding.astype(np.float32, copy=False))
    tmp_path.replace(output_path)


def _infer_batches(
    model: EsmModel,
    tokenizer: AutoTokenizer,
    samples: list[SampleRecord],
    device: str,
    batch_size: int,
    max_num_tokens: int | None,
) -> int:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    chain_tasks = [
        ChainTask(
            sample_id=sample.sample_id,
            chain_index=chain_index,
            chain_id=sample.chain_ids[chain_index],
            sequence=sequence,
        )
        for sample in samples
        for chain_index, sequence in enumerate(sample.chain_sequences)
    ]
    total_chains = {sample.sample_id: sample.chain_count for sample in samples}
    pending: dict[str, dict[int, np.ndarray]] = {}
    ok = 0

    with (
        torch.inference_mode(),
        tqdm(total=len(samples), desc="build_esm", unit="protein") as pbar,
    ):
        for batch in _iter_batches(
            chain_tasks, batch_size=batch_size, max_num_tokens=max_num_tokens
        ):
            sequences = [item.sequence for item in batch]
            toks = tokenizer(
                sequences, return_tensors="pt", padding=True, add_special_tokens=True
            )
            toks = {key: value.to(device) for key, value in toks.items()}
            outputs = model(**toks)

            hidden = outputs.last_hidden_state.detach().float().cpu()
            for idx, item in enumerate(batch):
                rep = hidden[idx, 1 : item.length + 1].numpy()
                if rep.shape[0] != item.length:
                    raise ValueError(
                        f"Unexpected embedding length for {item.sample_id}:{item.chain_id}: "
                        f"{rep.shape[0]} tokens for sequence length {item.length}"
                    )

                chain_embeddings = pending.setdefault(item.sample_id, {})
                chain_embeddings[item.chain_index] = rep
                if len(chain_embeddings) != total_chains[item.sample_id]:
                    continue

                sample = sample_by_id[item.sample_id]
                residue_embeddings = np.concatenate(
                    [
                        chain_embeddings[chain_index]
                        for chain_index in range(sample.chain_count)
                    ],
                    axis=0,
                )
                if residue_embeddings.shape[0] != sample.residue_count:
                    raise ValueError(
                        f"Unexpected residue count for {item.sample_id}: "
                        f"{residue_embeddings.shape[0]} embeddings for {sample.residue_count} residues"
                    )
                _write_embedding(sample.output_path, residue_embeddings)
                pending.pop(item.sample_id, None)
                ok += 1
                pbar.update(1)

    if pending:
        unfinished = ", ".join(sorted(pending))
        raise RuntimeError(f"Unfinished ESM samples remained in memory: {unfinished}")
    return ok


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="YAML config file.",
)
@click.option(
    "--path",
    "dataset_root",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--input-dir",
    default="protein_ligand",
    type=str,
    help="Sample directory under dataset root.",
)
@click.option(
    "--output-modal",
    default="esm",
    type=str,
    help="Output modality directory under dataset root.",
)
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.option(
    "--n-jobs",
    default=4,
    type=int,
    help="Worker processes for CPU-side sequence extraction.",
)
@click.option(
    "--batch-size",
    default=16,
    type=int,
    help="Maximum number of chain sequences per ESM forward pass.",
)
@click.option(
    "--max-num-tokens",
    default=None,
    type=int,
    help="Approximate padded token budget per ESM batch. Longer sequences are processed alone.",
)
@click.option(
    "--model-path",
    default=str(DEFAULT_HF_MODEL_ROOT),
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the local Hugging Face cache repo or resolved snapshot directory for facebook/esm2_t33_650M_UR50D.",
)
@click.option(
    "--overwrite/--skip-existing",
    default=False,
    help="Overwrite existing embedding files.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config: Path | None,
    dataset_root: Path,
    input_dir: str,
    output_modal: str,
    device: str,
    n_jobs: int,
    batch_size: int,
    max_num_tokens: int | None,
    model_path: Path,
    overwrite: bool,
):
    cfg = load_yaml_config(config)
    cfg_values = resolve_feature_build_config(cfg)

    def resolve(name: str, cli_value, cfg_value):
        source = ctx.get_parameter_source(name)
        if source != ParameterSource.DEFAULT:
            return cli_value
        return cfg_value if cfg_value is not None else cli_value

    dataset_root = resolve("dataset_root", dataset_root, cfg_values["dataset_root"])
    input_dir = resolve("input_dir", input_dir, cfg_values["input_dir"])
    output_modal = resolve("output_modal", output_modal, cfg_values["embedding_modal"])
    device = resolve("device", device, cfg_values["device"])
    n_jobs = resolve("n_jobs", n_jobs, cfg_values["n_jobs"])
    batch_size = resolve("batch_size", batch_size, cfg_values["batch_size"])
    max_num_tokens = resolve(
        "max_num_tokens", max_num_tokens, cfg_values["max_num_tokens"]
    )
    model_path = resolve(
        "model_path", model_path, cfg_values["model_path"] or DEFAULT_HF_MODEL_ROOT
    )
    overwrite = resolve("overwrite", overwrite, cfg_values["overwrite"])

    if dataset_root is None:
        raise click.ClickException(
            "Dataset root is required. Provide --path or set data.root + data.dataset_name in --config."
        )

    dataset_root = dataset_root.expanduser().resolve()
    sample_root = dataset_root / input_dir
    if not sample_root.exists():
        raise click.ClickException(f"Sample root does not exist: {sample_root}")

    output_dir = modal_dir(dataset_root, output_modal)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = _iter_sample_dirs(sample_root)
    sample_records = _load_samples(
        sample_dirs,
        dataset_root=dataset_root,
        output_modal=output_modal,
        overwrite=overwrite,
        n_jobs=n_jobs,
    )
    skipped = len(sample_dirs) - len(sample_records)
    if not sample_records:
        click.echo(f"done 0/0 (skipped_existing={skipped})")
        return

    resolved_model_path = _resolve_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path, local_files_only=True
    )
    model = EsmModel.from_pretrained(
        resolved_model_path,
        local_files_only=True,
        add_pooling_layer=False,
    )
    model.eval()

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dev = device
    model = model.to(device=dev, dtype=torch.float32)
    if max_num_tokens is None:
        max_num_tokens = 4096 if dev == "cuda" else 1024

    click.echo(f"Loaded Hugging Face ESM2 weights from: {resolved_model_path}")
    click.echo(
        "Chain-aware batch inference settings: "
        f"{len(sample_records)} proteins, "
        f"{sum(sample.chain_count for sample in sample_records)} chains, "
        f"batch_size={batch_size}, "
        f"max_num_tokens={max_num_tokens}, "
        f"skipped_existing={skipped}, "
        "dtype=torch.float32"
    )
    ok = _infer_batches(
        model=model,
        tokenizer=tokenizer,
        samples=sample_records,
        device=dev,
        batch_size=batch_size,
        max_num_tokens=max_num_tokens,
    )
    click.echo(f"done {ok}/{len(sample_records)}")


if __name__ == "__main__":
    main()
