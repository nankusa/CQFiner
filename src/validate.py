"""Re-evaluate UniSite validation with the original two-GPU batch grouping."""

import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.runtime import build_pocket_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("weights/cqfiner.pt"))
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    runtime = build_pocket_runtime(
        args.config,
        data_overrides={"root": str(args.data_root.resolve())},
        logger_version="validation",
        trainer_overrides={
            "enable_checkpointing": False,
            "enable_model_summary": False,
        },
        include_checkpoint_callbacks=False,
        include_periodic_test_callback=False,
        include_memory_monitor=False,
    )
    runtime.module.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)["state_dict"],
        strict=True,
    )
    result = runtime.trainer.validate(runtime.module, datamodule=runtime.data)
    if runtime.trainer.is_global_zero:
        path = Path(runtime.trainer.logger.log_dir) / "validation_metrics.json"
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
