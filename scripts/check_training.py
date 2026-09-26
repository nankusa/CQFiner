"""Two real-data Lightning optimizer steps; temporary outputs are deleted."""

import argparse
from pathlib import Path
import sys
import tempfile
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import UnifiedStructureDataset
from src.training.runtime import build_pocket_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ids = (ROOT / "data/splits/coach420/test_ids_coach420").read_text().split()[:2]
    dataset = UnifiedStructureDataset(
        args.data_root / "coach420", ids, "residue", "esm", 1280, ["esm"]
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        follow_batch=["target_pos", "ligand_pos", "target_mask_host_id", "surface_pos"],
    )
    with tempfile.TemporaryDirectory(prefix="cqfiner_training_check_") as tmp:
        runtime = build_pocket_runtime(
            ROOT / "configs/train.yaml",
            data_overrides={"root": str(args.data_root.resolve())},
            trainer_overrides={
                "accelerator": "cpu",
                "devices": 1,
                "strategy": "auto",
                "precision": "32-true",
                "max_steps": 2,
                "max_epochs": 1,
                "limit_train_batches": 2,
                "limit_val_batches": 1,
                "num_sanity_val_steps": 0,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "log_every_n_steps": 1,
                "logger": {
                    "save_dir": tmp,
                    "name": "train",
                    "version": "check",
                    "default_hp_metric": False,
                },
            },
            include_checkpoint_callbacks=False,
            include_periodic_test_callback=False,
            include_memory_monitor=False,
        )
        before = runtime.module.model.input_proj.weight.detach().clone()
        runtime.trainer.fit(
            runtime.module, train_dataloaders=loader, val_dataloaders=loader
        )
        if runtime.trainer.global_step != 2 or torch.equal(
            before, runtime.module.model.input_proj.weight
        ):
            raise RuntimeError("Optimizer did not update parameters")
        saved = Path(tmp) / "check.ckpt"
        runtime.trainer.save_checkpoint(saved)
        runtime.module.load_state_dict(
            torch.load(saved, map_location="cpu", weights_only=False)["state_dict"],
            strict=True,
        )
        if not all(torch.isfinite(p).all() for p in runtime.module.parameters()):
            raise FloatingPointError("Nonfinite trained parameters")
        runtime.trainer.logger.experiment.close()
    print(
        "PASS: full-architecture training, validation and strict checkpoint reload; temporary artifacts removed."
    )


if __name__ == "__main__":
    main()
