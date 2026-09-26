from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


BYTES_PER_GIB = 1024**3


def _batch_num_graphs(batch: Any) -> int:
    if not hasattr(batch, "num_graphs"):
        raise TypeError(
            f"Expected a PyG batch with num_graphs, got {type(batch).__name__}."
        )
    num_graphs = int(batch.num_graphs)
    if num_graphs <= 0:
        raise ValueError(f"batch.num_graphs must be positive, got {num_graphs}.")
    return num_graphs


class ComplexityRecorder(Callback):
    def __init__(self, *, stages: set[str]) -> None:
        invalid = set(stages) - {"train", "test"}
        if invalid:
            raise ValueError(f"Unsupported complexity stage(s): {sorted(invalid)}.")
        if not stages:
            raise ValueError("ComplexityRecorder requires at least one stage.")
        self.stages = set(stages)
        self.records: dict[str, list[dict[str, float | int]]] = {}
        self._batch_start: dict[str, float] = {}

    @staticmethod
    def _cuda_device(pl_module: L.LightningModule) -> torch.device:
        device = pl_module.device
        if device.type != "cuda":
            raise RuntimeError(
                f"Complexity profiling requires a CUDA device, got {device}."
            )
        return device

    @staticmethod
    def _assert_single_device(trainer: L.Trainer) -> None:
        num_devices = int(getattr(trainer, "num_devices", 1))
        num_nodes = int(getattr(trainer, "num_nodes", 1))
        if num_devices != 1 or num_nodes != 1:
            raise NotImplementedError(
                "Complexity profiling currently requires a single process and a single CUDA device."
            )

    def _sync(self, pl_module: L.LightningModule) -> torch.device:
        device = self._cuda_device(pl_module)
        torch.cuda.synchronize(device)
        return device

    def _dataset_name(self, trainer: L.Trainer, dataloader_idx: int) -> str:
        datamodule = getattr(trainer, "datamodule", None)
        indices = (
            getattr(datamodule, "test_dataloader_indices", {})
            if datamodule is not None
            else {}
        )
        for name, idx in indices.items():
            if int(idx) == int(dataloader_idx):
                return str(name)
        raise KeyError(
            f"Could not resolve test dataloader_idx={dataloader_idx} to a dataset name."
        )

    def _start_batch(self, key: str, pl_module: L.LightningModule) -> None:
        device = self._sync(pl_module)
        torch.cuda.reset_peak_memory_stats(device)
        if key in self._batch_start:
            raise RuntimeError(f"Complexity key {key!r} started twice before ending.")
        self._batch_start[key] = time.perf_counter()

    def _end_batch(self, key: str, pl_module: L.LightningModule, batch: Any) -> None:
        device = self._sync(pl_module)
        start = self._batch_start.pop(key, None)
        if start is None:
            raise RuntimeError(f"Complexity key {key!r} ended before starting.")
        num_graphs = _batch_num_graphs(batch)
        peak_allocated_gib = (
            float(torch.cuda.max_memory_allocated(device)) / BYTES_PER_GIB
        )
        peak_reserved_gib = (
            float(torch.cuda.max_memory_reserved(device)) / BYTES_PER_GIB
        )
        self.records.setdefault(key, []).append(
            {
                "step_seconds": time.perf_counter() - start,
                "num_graphs": num_graphs,
                "peak_allocated_gib": peak_allocated_gib,
                "peak_allocated_gib_per_sample": peak_allocated_gib / float(num_graphs),
                "peak_reserved_gib": peak_reserved_gib,
                "peak_reserved_gib_per_sample": peak_reserved_gib / float(num_graphs),
            }
        )

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if "train" in self.stages:
            self._assert_single_device(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if "test" in self.stages:
            self._assert_single_device(trainer)

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if "train" not in self.stages:
            return
        self._start_batch("train", pl_module)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if "train" not in self.stages:
            return
        self._end_batch("train", pl_module, batch)

    def on_test_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "test" not in self.stages:
            return
        dataset_name = self._dataset_name(trainer, dataloader_idx)
        self._start_batch(f"test/{dataset_name}", pl_module)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "test" not in self.stages:
            return
        dataset_name = self._dataset_name(trainer, dataloader_idx)
        self._end_batch(f"test/{dataset_name}", pl_module, batch)

    def summary_rows(self) -> list[dict[str, float | int | str]]:
        if not self.records:
            raise RuntimeError("ComplexityRecorder has no profiled batches.")
        return [
            self._summarize_key(key, records)
            for key, records in sorted(self.records.items())
        ]

    def metric_dict(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for row in self.summary_rows():
            prefix = f"{row['stage']}/profile"
            for key, value in row.items():
                if key == "stage":
                    continue
                metrics[f"{prefix}/{key}"] = float(value)
        return metrics

    @staticmethod
    def _summarize_key(
        key: str, records: list[dict[str, float | int]]
    ) -> dict[str, float | int | str]:
        if not records:
            raise RuntimeError(f"No complexity records for {key}.")
        num_steps = len(records)
        num_samples = sum(int(record["num_graphs"]) for record in records)
        if num_samples <= 0:
            raise RuntimeError(f"No profiled samples for {key}.")
        step_seconds = [float(record["step_seconds"]) for record in records]
        total_step_seconds = sum(step_seconds)
        if total_step_seconds <= 0.0:
            raise RuntimeError(
                f"Total profiled step duration for {key} must be positive."
            )
        peak_allocated = [float(record["peak_allocated_gib"]) for record in records]
        peak_allocated_per_sample = [
            float(record["peak_allocated_gib_per_sample"]) for record in records
        ]
        peak_reserved = [float(record["peak_reserved_gib"]) for record in records]
        peak_reserved_per_sample = [
            float(record["peak_reserved_gib_per_sample"]) for record in records
        ]
        return {
            "stage": key,
            "num_steps": num_steps,
            "num_samples": num_samples,
            "mean_batch_size": num_samples / float(num_steps),
            "total_step_seconds": total_step_seconds,
            "seconds_per_sample": total_step_seconds / float(num_samples),
            "samples_per_second": float(num_samples) / total_step_seconds,
            "mean_step_seconds": total_step_seconds / float(num_steps),
            "mean_peak_allocated_gib": sum(peak_allocated) / float(num_steps),
            "mean_peak_allocated_gib_per_sample": sum(peak_allocated_per_sample)
            / float(num_steps),
            "max_peak_allocated_gib": max(peak_allocated),
            "max_peak_allocated_gib_per_sample": max(peak_allocated_per_sample),
            "mean_peak_reserved_gib": sum(peak_reserved) / float(num_steps),
            "mean_peak_reserved_gib_per_sample": sum(peak_reserved_per_sample)
            / float(num_steps),
            "max_peak_reserved_gib": max(peak_reserved),
            "max_peak_reserved_gib_per_sample": max(peak_reserved_per_sample),
        }

    def write(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = self.summary_rows()
        path = output / "complexity_summary.csv"
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class EpochTimeRecorder(Callback):
    def __init__(self, *, stages: set[str]) -> None:
        invalid = set(stages) - {"train", "test"}
        if invalid:
            raise ValueError(f"Unsupported timing stage(s): {sorted(invalid)}.")
        if not stages:
            raise ValueError("EpochTimeRecorder requires at least one stage.")
        self.stages = set(stages)
        self.records: dict[str, dict[str, float | int | None]] = {}
        self._active_key: str | None = None

    @staticmethod
    def _cuda_device(pl_module: L.LightningModule) -> torch.device:
        device = pl_module.device
        if device.type != "cuda":
            raise RuntimeError(f"Epoch timing requires a CUDA device, got {device}.")
        return device

    @staticmethod
    def _assert_single_device(trainer: L.Trainer) -> None:
        num_devices = int(getattr(trainer, "num_devices", 1))
        num_nodes = int(getattr(trainer, "num_nodes", 1))
        if num_devices != 1 or num_nodes != 1:
            raise NotImplementedError(
                "Epoch timing currently requires a single process and a single CUDA device."
            )

    @staticmethod
    def _dataset_name(trainer: L.Trainer, dataloader_idx: int) -> str:
        datamodule = getattr(trainer, "datamodule", None)
        indices = (
            getattr(datamodule, "test_dataloader_indices", {})
            if datamodule is not None
            else {}
        )
        for name, idx in indices.items():
            if int(idx) == int(dataloader_idx):
                return str(name)
        raise KeyError(
            f"Could not resolve test dataloader_idx={dataloader_idx} to a dataset name."
        )

    @staticmethod
    def _num_test_batches(trainer: L.Trainer, dataloader_idx: int) -> int | None:
        num_batches = getattr(trainer, "num_test_batches", None)
        if num_batches is None:
            return None
        if isinstance(num_batches, int):
            return int(num_batches)
        if isinstance(num_batches, (list, tuple)):
            if dataloader_idx >= len(num_batches):
                raise IndexError(
                    f"num_test_batches has length {len(num_batches)}, cannot read dataloader_idx={dataloader_idx}."
                )
            value = num_batches[dataloader_idx]
            if value == float("inf"):
                return None
            return int(value)
        raise TypeError(
            f"Unexpected trainer.num_test_batches type: {type(num_batches).__name__}."
        )

    @staticmethod
    def _num_train_batches(trainer: L.Trainer) -> int | None:
        num_batches = getattr(trainer, "num_training_batches", None)
        if num_batches is None or num_batches == float("inf"):
            return None
        if isinstance(num_batches, int):
            return int(num_batches)
        raise TypeError(
            f"Unexpected trainer.num_training_batches type: {type(num_batches).__name__}."
        )

    def _start_key(self, key: str, pl_module: L.LightningModule) -> None:
        if self._active_key is not None:
            raise RuntimeError(
                f"Cannot start timing key {key!r}; active key is {self._active_key!r}."
            )
        if key in self.records:
            raise RuntimeError(f"Timing key {key!r} was reopened after it was closed.")
        device = self._cuda_device(pl_module)
        torch.cuda.synchronize(device)
        self.records[key] = {
            "start_time": time.perf_counter(),
            "total_seconds": None,
            "num_steps": 0,
            "num_samples": 0,
        }
        self._active_key = key

    def _close_active(self, pl_module: L.LightningModule) -> None:
        if self._active_key is None:
            return
        device = self._cuda_device(pl_module)
        torch.cuda.synchronize(device)
        record = self.records[self._active_key]
        start = record["start_time"]
        if start is None:
            raise RuntimeError(
                f"Inference timing key {self._active_key!r} has no start_time."
            )
        record["total_seconds"] = time.perf_counter() - float(start)
        self._active_key = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if "train" in self.stages:
            self._assert_single_device(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if "test" in self.stages:
            self._assert_single_device(trainer)

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if "train" not in self.stages:
            return
        if self._active_key is None:
            self._start_key("train", pl_module)
        elif self._active_key != "train":
            raise RuntimeError(
                f"Train timing started while active key is {self._active_key!r}."
            )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if "train" not in self.stages:
            return
        if self._active_key != "train":
            raise RuntimeError(
                f"Train timing ended batch while active key is {self._active_key!r}."
            )
        record = self.records["train"]
        record["num_steps"] = int(record["num_steps"]) + 1
        record["num_samples"] = int(record["num_samples"]) + _batch_num_graphs(batch)
        expected_batches = self._num_train_batches(trainer)
        if expected_batches is not None and int(batch_idx) + 1 == expected_batches:
            self._close_active(pl_module)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if "train" in self.stages and self._active_key == "train":
            self._close_active(pl_module)

    def on_test_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "test" not in self.stages:
            return
        key = f"test/{self._dataset_name(trainer, dataloader_idx)}"
        if self._active_key == key:
            return
        self._close_active(pl_module)
        self._start_key(key, pl_module)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "test" not in self.stages:
            return
        key = f"test/{self._dataset_name(trainer, dataloader_idx)}"
        if self._active_key != key:
            raise RuntimeError(
                f"Test timing ended batch for {key!r} while active key is {self._active_key!r}."
            )
        record = self.records[key]
        record["num_steps"] = int(record["num_steps"]) + 1
        record["num_samples"] = int(record["num_samples"]) + _batch_num_graphs(batch)
        expected_batches = self._num_test_batches(trainer, dataloader_idx)
        if expected_batches is not None and int(batch_idx) + 1 == expected_batches:
            self._close_active(pl_module)

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if "test" in self.stages:
            self._close_active(pl_module)

    def summary_rows(self) -> list[dict[str, float | int | str]]:
        if not self.records:
            raise RuntimeError("EpochTimeRecorder has no timed batches.")
        rows = []
        for key, record in sorted(self.records.items()):
            total_seconds = record["total_seconds"]
            if total_seconds is None:
                raise RuntimeError(f"Timing key {key!r} was not closed.")
            num_steps = int(record["num_steps"])
            num_samples = int(record["num_samples"])
            if num_steps <= 0:
                raise RuntimeError(f"Timing key {key!r} has no timed steps.")
            if num_samples <= 0:
                raise RuntimeError(f"Timing key {key!r} has no timed samples.")
            total_seconds = float(total_seconds)
            if total_seconds <= 0.0:
                raise RuntimeError(f"Timing key {key!r} duration must be positive.")
            rows.append(
                {
                    "stage": key,
                    "num_steps": num_steps,
                    "num_samples": num_samples,
                    "mean_batch_size": num_samples / float(num_steps),
                    "total_seconds": total_seconds,
                    "seconds_per_sample": total_seconds / float(num_samples),
                    "samples_per_second": float(num_samples) / total_seconds,
                }
            )
        return rows

    def metric_dict(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for row in self.summary_rows():
            prefix = f"{row['stage']}/time"
            for key, value in row.items():
                if key == "stage":
                    continue
                metrics[f"{prefix}/{key}"] = float(value)
        return metrics


class InferenceTimeRecorder(EpochTimeRecorder):
    def __init__(self) -> None:
        super().__init__(stages={"test"})


class StageTimingRecorder(Callback):
    def __init__(self) -> None:
        self.records: dict[str, list[dict[str, float | int]]] = {}

    @staticmethod
    def _assert_single_device(trainer: L.Trainer) -> None:
        num_devices = int(getattr(trainer, "num_devices", 1))
        num_nodes = int(getattr(trainer, "num_nodes", 1))
        if num_devices != 1 or num_nodes != 1:
            raise NotImplementedError(
                "Stage timing currently requires a single process and a single CUDA device."
            )

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._assert_single_device(trainer)
        existing = getattr(pl_module, "eval_stage_timing_recorder", None)
        if existing is not None:
            raise RuntimeError(
                "pl_module already has eval_stage_timing_recorder attached."
            )
        pl_module.eval_stage_timing_recorder = self

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        existing = getattr(pl_module, "eval_stage_timing_recorder", None)
        if existing is not self:
            raise RuntimeError("StageTimingRecorder detach mismatch.")
        delattr(pl_module, "eval_stage_timing_recorder")

    def record(
        self, *, dataset_name: str, stage: str, seconds: float, num_graphs: int
    ) -> None:
        seconds = float(seconds)
        num_graphs = int(num_graphs)
        if seconds <= 0.0:
            raise ValueError(
                f"Stage timing seconds must be positive for {stage}, got {seconds}."
            )
        if num_graphs <= 0:
            raise ValueError(
                f"Stage timing num_graphs must be positive for {stage}, got {num_graphs}."
            )
        key = f"test/{dataset_name}/{stage}"
        self.records.setdefault(key, []).append(
            {
                "step_seconds": seconds,
                "num_graphs": num_graphs,
            }
        )

    def summary_rows(self) -> list[dict[str, float | int | str]]:
        if not self.records:
            raise RuntimeError("StageTimingRecorder has no timed stages.")
        rows: list[dict[str, float | int | str]] = []
        for key, records in sorted(self.records.items()):
            num_steps = len(records)
            num_samples = sum(int(record["num_graphs"]) for record in records)
            if num_steps <= 0:
                raise RuntimeError(f"Stage timing key {key!r} has no timed steps.")
            if num_samples <= 0:
                raise RuntimeError(f"Stage timing key {key!r} has no timed samples.")
            total_seconds = sum(float(record["step_seconds"]) for record in records)
            if total_seconds <= 0.0:
                raise RuntimeError(
                    f"Stage timing key {key!r} duration must be positive."
                )
            rows.append(
                {
                    "stage": key,
                    "num_steps": num_steps,
                    "num_samples": num_samples,
                    "mean_batch_size": num_samples / float(num_steps),
                    "total_seconds": total_seconds,
                    "seconds_per_sample": total_seconds / float(num_samples),
                    "samples_per_second": float(num_samples) / total_seconds,
                    "mean_step_seconds": total_seconds / float(num_steps),
                }
            )
        return rows

    def metric_dict(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for row in self.summary_rows():
            prefix = f"{row['stage']}/stage_time"
            for key, value in row.items():
                if key == "stage":
                    continue
                metrics[f"{prefix}/{key}"] = float(value)
        return metrics
