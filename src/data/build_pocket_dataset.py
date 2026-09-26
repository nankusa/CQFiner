#!/usr/bin/env python3
"""Run the full feature-building pipeline for one dataset root."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import click
from click.core import ParameterSource

from .config_utils import load_yaml_config, resolve_feature_build_config


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
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.option("--n-jobs", default=1, type=int)
@click.option("--threshold", default=4.0, type=float)
@click.option("--surface-sasa-threshold", default=1e-4, type=float)
@click.option(
    "--use-residue-depths/--no-use-residue-depths",
    default=None,
    help="Whether to compute residue depths via Bio.PDB.ResidueDepth/MSMS during pocket target building.",
)
@click.option("--embedding-modal", default="esm", type=str)
@click.option("--target-modal", default="pocket", type=str)
@click.option(
    "--batch-size",
    default=16,
    type=int,
    help="Maximum number of chain sequences in one ESM forward pass.",
)
@click.option(
    "--max-num-tokens",
    default=None,
    type=int,
    help="Approximate padded token budget per ESM batch.",
)
@click.option(
    "--model-path", default=None, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--overwrite/--skip-existing", default=False)
@click.pass_context
def main(
    ctx: click.Context,
    config: Path | None,
    dataset_root: Path,
    input_dir: str,
    device: str,
    n_jobs: int,
    threshold: float,
    surface_sasa_threshold: float,
    use_residue_depths: bool | None,
    embedding_modal: str,
    target_modal: str,
    batch_size: int,
    max_num_tokens: int | None,
    model_path: Path | None,
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
    device = resolve("device", device, cfg_values["device"])
    n_jobs = resolve("n_jobs", n_jobs, cfg_values["n_jobs"])
    threshold = resolve("threshold", threshold, cfg_values["threshold"])
    surface_sasa_threshold = resolve(
        "surface_sasa_threshold",
        surface_sasa_threshold,
        cfg_values["surface_sasa_threshold"],
    )
    use_residue_depths = resolve(
        "use_residue_depths", use_residue_depths, cfg_values["use_residue_depths"]
    )
    embedding_modal = resolve(
        "embedding_modal", embedding_modal, cfg_values["embedding_modal"]
    )
    target_modal = resolve("target_modal", target_modal, cfg_values["target_modal"])
    batch_size = resolve("batch_size", batch_size, cfg_values["batch_size"])
    max_num_tokens = resolve(
        "max_num_tokens", max_num_tokens, cfg_values["max_num_tokens"]
    )
    model_path = resolve("model_path", model_path, cfg_values["model_path"])
    overwrite = resolve("overwrite", overwrite, cfg_values["overwrite"])

    if dataset_root is None:
        raise click.ClickException(
            "Dataset root is required. Provide --path or set data.root + data.dataset_name in --config."
        )

    dataset_root = dataset_root.expanduser().resolve()

    esm_command = [
        sys.executable,
        "-m",
        "src.data.build_esm_embeddings",
        "--path",
        str(dataset_root),
        "--input-dir",
        input_dir,
        "--output-modal",
        embedding_modal,
        "--device",
        device,
        "--n-jobs",
        str(n_jobs),
        "--batch-size",
        str(batch_size),
    ]
    if max_num_tokens is not None:
        esm_command.extend(["--max-num-tokens", str(max_num_tokens)])
    if model_path is not None:
        esm_command.extend(["--model-path", str(model_path)])
    esm_command.append("--overwrite" if overwrite else "--skip-existing")

    binding_command = [
        sys.executable,
        "-m",
        "src.data.build_binding_info",
        "--path",
        str(dataset_root),
        "--input-dir",
        input_dir,
        "--output-modal",
        target_modal,
        "--threshold",
        str(threshold),
        "--surface-sasa-threshold",
        str(surface_sasa_threshold),
        "--use-residue-depths" if use_residue_depths else "--no-use-residue-depths",
        "--n-jobs",
        str(n_jobs),
    ]
    binding_command.append("--overwrite" if overwrite else "--skip-existing")

    subprocess.check_call(esm_command)
    subprocess.check_call(binding_command)


if __name__ == "__main__":
    main()
