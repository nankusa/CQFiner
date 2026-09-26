from __future__ import annotations

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm


class PeriodicTestCallback(Callback):
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = int(every_n_epochs)

    @staticmethod
    def _normalize_loaders(loaders) -> list:
        if loaders is None:
            return []
        if isinstance(loaders, list):
            return loaders
        if isinstance(loaders, tuple):
            return list(loaders)
        return [loaders]

    @staticmethod
    def _distributed_enabled() -> bool:
        return (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )

    def _build_test_loaders(self, trainer) -> list:
        datamodule = trainer.datamodule
        default_loaders = self._normalize_loaders(datamodule.test_dataloader())
        if not self._distributed_enabled():
            return default_loaders

        if not hasattr(datamodule, "_resolve_dataset_cfg") or not hasattr(
            datamodule, "_dataset"
        ):
            return default_loaders

        follow_batch = [
            "target_pos",
            "ligand_pos",
            "target_mask_host_id",
            "surface_pos",
        ]
        loaders = []
        for cfg in getattr(datamodule, "test_datasets", []):
            resolved = datamodule._resolve_dataset_cfg(cfg)
            dataset = datamodule._dataset("test", cfg=resolved)
            sampler = DistributedSampler(dataset, shuffle=False)
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=resolved["batch_size"],
                    shuffle=False,
                    sampler=sampler,
                    num_workers=resolved["num_workers"],
                    pin_memory=resolved["pin_memory"],
                    follow_batch=follow_batch,
                )
            )
        return loaders

    @staticmethod
    def _dataset_name(trainer, dataloader_idx: int) -> str:
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return f"loader_{dataloader_idx}"
        indices = getattr(datamodule, "test_dataloader_indices", {})
        for name, idx in indices.items():
            if idx == dataloader_idx:
                return str(name)
        return f"loader_{dataloader_idx}"

    def _should_run(self, trainer) -> bool:
        if self.every_n_epochs <= 0:
            return False
        if trainer.sanity_checking:
            return False
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return False
        if not getattr(datamodule, "test_datasets", None):
            return False
        return (trainer.current_epoch + 1) % self.every_n_epochs == 0

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._should_run(trainer):
            return

        loaders = self._build_test_loaders(trainer)
        if not loaders:
            return

        if hasattr(trainer.strategy, "barrier"):
            trainer.strategy.barrier("periodic_test_start")

        was_training = pl_module.training
        pl_module.on_test_epoch_start()
        pl_module.eval()

        with torch.inference_mode():
            for dataloader_idx, loader in enumerate(loaders):
                sampler = getattr(loader, "sampler", None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(trainer.current_epoch)

                dataset_name = self._dataset_name(trainer, dataloader_idx)
                batch_iter = loader
                progress = None
                if trainer.is_global_zero:
                    total = len(loader) if hasattr(loader, "__len__") else None
                    progress = tqdm(
                        loader,
                        total=total,
                        desc=f"Test DataLoader {dataloader_idx} [{dataset_name}]",
                        dynamic_ncols=True,
                    )
                    batch_iter = progress

                for batch_idx, batch in enumerate(batch_iter):
                    batch = batch.to(pl_module.device)
                    pl_module.test_step(batch, batch_idx, dataloader_idx=dataloader_idx)

                if progress is not None:
                    progress.close()

        pl_module.on_test_epoch_end()

        if was_training:
            pl_module.train()

        if hasattr(trainer.strategy, "barrier"):
            trainer.strategy.barrier("periodic_test_end")
