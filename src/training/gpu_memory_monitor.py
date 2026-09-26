from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


@dataclass(frozen=True)
class GpuMemorySnapshot:
    used_mb: float
    total_mb: float
    backend: str
    allocated_mb: float | None = None
    reserved_mb: float | None = None
    max_allocated_mb: float | None = None
    max_reserved_mb: float | None = None


class GpuMemoryMonitor(Callback):
    """Low-overhead GPU memory monitor.

    The default backend uses CUDA/PyTorch APIs in-process, avoiding the large
    overhead of spawning nvidia-smi during training. If pynvml is installed, the
    ``nvml`` backend can be selected to read device-level memory through NVML.
    """

    def __init__(
        self,
        enabled: bool = True,
        poll_interval_seconds: float = 1.0,
        sample_every_n_steps: int = 50,
        sync_cuda: bool = False,
        sample_on_batch_hooks: bool = False,
        backend: str = "auto",
        log_prefix: str = "memory",
        prog_bar: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.poll_interval_seconds = float(max(poll_interval_seconds, 0.0))
        self.sample_every_n_steps = int(max(sample_every_n_steps, 1))
        self.sync_cuda = bool(sync_cuda)
        self.sample_on_batch_hooks = bool(sample_on_batch_hooks)
        self.backend = str(backend).lower()
        if self.backend not in {"torch", "nvml", "auto"}:
            raise ValueError(f"Unsupported GPU memory monitor backend: {backend}")
        self.log_prefix = str(log_prefix).strip("/") or "memory"
        self.prog_bar = bool(prog_bar)

        self._selector: str | None = None
        self._local_device_index: int | None = None
        self._pynvml: Any | None = None
        self._nvml_init_attempted = False
        self._last_snapshot: GpuMemorySnapshot | None = None
        self._run_max_mb = 0.0
        self._epoch_max_mb = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._warned_unavailable = False
        self._warned_query = False

    def setup(self, trainer, pl_module, stage: str) -> None:  # type: ignore[override]
        del trainer, pl_module, stage
        self._selector = self._resolve_current_gpu_selector()

    def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        del trainer, pl_module
        self._start_polling()
        self._sample()

    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        del trainer, pl_module
        self._sample()
        self._stop_polling()

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        del trainer, pl_module
        with self._lock:
            self._epoch_max_mb = 0.0
        self._sample()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:  # type: ignore[override]
        del pl_module, batch
        if self.sample_on_batch_hooks and self._should_sample(trainer, batch_idx):
            self._sample()

    def on_before_backward(self, trainer, pl_module, loss) -> None:  # type: ignore[override]
        del pl_module, loss
        if self.sample_on_batch_hooks and self._should_sample(
            trainer, int(getattr(trainer, "global_step", 0))
        ):
            self._sample()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:  # type: ignore[override]
        del pl_module, optimizer
        if self.sample_on_batch_hooks and self._should_sample(
            trainer, int(getattr(trainer, "global_step", 0))
        ):
            self._sample()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:  # type: ignore[override]
        del outputs, batch
        if self._should_sample(trainer, batch_idx):
            if self.sample_on_batch_hooks:
                self._sample()
            self._log_snapshot(pl_module, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # type: ignore[override]
        del outputs, batch, dataloader_idx
        if self._should_sample(trainer, batch_idx):
            if self.sample_on_batch_hooks:
                self._sample()
            self._log_snapshot(pl_module, on_step=True, on_epoch=False)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:  # type: ignore[override]
        del outputs, batch, dataloader_idx
        if self._should_sample(trainer, batch_idx):
            if self.sample_on_batch_hooks:
                self._sample()
            self._log_snapshot(pl_module, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        del trainer
        self._sample()
        self._log_snapshot(pl_module, on_step=False, on_epoch=True)

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:  # type: ignore[override]
        del trainer, pl_module, exception
        self._sample()
        self._stop_polling()

    def _start_polling(self) -> None:
        if (
            not self.enabled
            or self.poll_interval_seconds <= 0.0
            or self._poll_thread is not None
        ):
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="gpu-memory-monitor", daemon=True
        )
        self._poll_thread.start()

    def _stop_polling(self) -> None:
        self._stop_event.set()
        thread = self._poll_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval_seconds):
            self._sample()

    def _should_sample(self, trainer, batch_idx: int) -> bool:
        del trainer
        return self.enabled and (int(batch_idx) % self.sample_every_n_steps == 0)

    def _resolve_current_gpu_selector(self) -> str | None:
        if not torch.cuda.is_available():
            return None
        local_idx = int(torch.cuda.current_device())
        self._local_device_index = local_idx
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            tokens = [token.strip() for token in visible.split(",") if token.strip()]
            if 0 <= local_idx < len(tokens):
                return tokens[local_idx]
        return str(local_idx)

    def _sample(self) -> None:
        snapshot = self._query_current_gpu_memory()
        if snapshot is None:
            return
        with self._lock:
            self._last_snapshot = snapshot
            self._run_max_mb = max(self._run_max_mb, snapshot.used_mb)
            self._epoch_max_mb = max(self._epoch_max_mb, snapshot.used_mb)

    def _query_current_gpu_memory(self) -> GpuMemorySnapshot | None:
        if not self.enabled:
            return None
        if not torch.cuda.is_available():
            self._warn_unavailable_once(
                "CUDA is not available; GPU memory monitor is disabled."
            )
            return None
        if self.sync_cuda:
            try:
                torch.cuda.synchronize(self._local_device_index)
            except Exception:
                pass

        if self.backend in {"auto", "nvml"}:
            snapshot = self._query_nvml_gpu_memory()
            if snapshot is not None or self.backend == "nvml":
                return snapshot
        return self._query_torch_gpu_memory()

    def _torch_allocator_metrics(self) -> dict[str, float]:
        device = self._local_device_index
        return {
            "allocated_mb": float(torch.cuda.memory_allocated(device)) / (1024.0**2),
            "reserved_mb": float(torch.cuda.memory_reserved(device)) / (1024.0**2),
            "max_allocated_mb": float(torch.cuda.max_memory_allocated(device))
            / (1024.0**2),
            "max_reserved_mb": float(torch.cuda.max_memory_reserved(device))
            / (1024.0**2),
        }

    def _query_torch_gpu_memory(self) -> GpuMemorySnapshot | None:
        device = self._local_device_index
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception as exc:
            if not self._warned_query:
                self._warned_query = True
                rank_zero_warn(
                    f"Failed to query CUDA memory with torch.cuda.mem_get_info: {exc}"
                )
            return None
        used_mb = float(total_bytes - free_bytes) / (1024.0**2)
        total_mb = float(total_bytes) / (1024.0**2)
        return GpuMemorySnapshot(
            used_mb=used_mb,
            total_mb=total_mb,
            backend="torch",
            **self._torch_allocator_metrics(),
        )

    def _init_nvml(self) -> bool:
        if self._pynvml is not None:
            return True
        if self._nvml_init_attempted:
            return False
        self._nvml_init_attempted = True
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._pynvml = pynvml
            return True
        except Exception as exc:
            if self.backend == "nvml":
                self._warn_unavailable_once(
                    f"NVML backend requested but pynvml is unavailable: {exc}"
                )
            return False

    def _query_nvml_gpu_memory(self) -> GpuMemorySnapshot | None:
        if not self._init_nvml() or self._pynvml is None:
            return None
        selector = self._selector or self._resolve_current_gpu_selector()
        if selector is None:
            self._warn_unavailable_once(
                "Could not resolve the current CUDA device for GPU memory monitoring."
            )
            return None

        try:
            if selector.isdigit():
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(int(selector))
            else:
                try:
                    handle = self._pynvml.nvmlDeviceGetHandleByUUID(selector)
                except TypeError:
                    handle = self._pynvml.nvmlDeviceGetHandleByUUID(selector.encode())
            info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception as exc:
            if not self._warned_query:
                self._warned_query = True
                rank_zero_warn(f"Failed to query NVML GPU memory: {exc}")
            return None

        return GpuMemorySnapshot(
            used_mb=float(info.used) / (1024.0**2),
            total_mb=float(info.total) / (1024.0**2),
            backend="nvml",
            **self._torch_allocator_metrics(),
        )

    def _log_snapshot(self, pl_module, on_step: bool, on_epoch: bool) -> None:
        with self._lock:
            snapshot = self._last_snapshot
            run_max_mb = self._run_max_mb
            epoch_max_mb = self._epoch_max_mb
        if snapshot is None:
            return

        used_gb = snapshot.used_mb / 1024.0
        total_gb = snapshot.total_mb / 1024.0
        run_max_gb = run_max_mb / 1024.0
        epoch_max_gb = epoch_max_mb / 1024.0
        used_pct = 100.0 * snapshot.used_mb / max(snapshot.total_mb, 1.0)
        prefix = self.log_prefix

        log_kwargs = {
            "on_step": on_step,
            "on_epoch": on_epoch,
            "prog_bar": self.prog_bar,
            "sync_dist": True,
            "reduce_fx": "max",
        }
        pl_module.log(f"{prefix}/nvidia_smi_used_gb", used_gb, **log_kwargs)
        pl_module.log(f"{prefix}/nvidia_smi_used_mb", snapshot.used_mb, **log_kwargs)
        pl_module.log(f"{prefix}/nvidia_smi_used_pct", used_pct, **log_kwargs)
        pl_module.log(f"{prefix}/nvidia_smi_total_gb", total_gb, **log_kwargs)
        pl_module.log(f"{prefix}/nvidia_smi_run_max_gb", run_max_gb, **log_kwargs)
        pl_module.log(f"{prefix}/nvidia_smi_epoch_max_gb", epoch_max_gb, **log_kwargs)
        pl_module.log(f"{prefix}/gpu_used_gb", used_gb, **log_kwargs)
        pl_module.log(f"{prefix}/gpu_used_mb", snapshot.used_mb, **log_kwargs)
        pl_module.log(f"{prefix}/gpu_used_pct", used_pct, **log_kwargs)
        pl_module.log(f"{prefix}/gpu_total_gb", total_gb, **log_kwargs)
        pl_module.log(f"{prefix}/gpu_run_max_gb", run_max_gb, **log_kwargs)
        pl_module.log(f"{prefix}/gpu_epoch_max_gb", epoch_max_gb, **log_kwargs)
        if snapshot.allocated_mb is not None:
            pl_module.log(
                f"{prefix}/torch_allocated_gb",
                snapshot.allocated_mb / 1024.0,
                **log_kwargs,
            )
        if snapshot.reserved_mb is not None:
            pl_module.log(
                f"{prefix}/torch_reserved_gb",
                snapshot.reserved_mb / 1024.0,
                **log_kwargs,
            )
        if snapshot.max_allocated_mb is not None:
            pl_module.log(
                f"{prefix}/torch_max_allocated_gb",
                snapshot.max_allocated_mb / 1024.0,
                **log_kwargs,
            )
        if snapshot.max_reserved_mb is not None:
            pl_module.log(
                f"{prefix}/torch_max_reserved_gb",
                snapshot.max_reserved_mb / 1024.0,
                **log_kwargs,
            )

    def _warn_unavailable_once(self, message: str) -> None:
        if self._warned_unavailable:
            return
        self._warned_unavailable = True
        rank_zero_warn(message)


NvidiaSmiMemoryMonitor = GpuMemoryMonitor


def build_gpu_memory_monitor(
    trainer_cfg: Mapping[str, Any] | None,
) -> GpuMemoryMonitor | None:
    cfg = dict(trainer_cfg or {})
    raw_enabled = cfg.get("monitor_gpu_memory", True)
    monitor_cfg: dict[str, Any] = {}
    if isinstance(raw_enabled, Mapping):
        monitor_cfg.update(dict(raw_enabled))
        enabled = bool(monitor_cfg.pop("enabled", True))
    else:
        enabled = bool(raw_enabled)
    monitor_cfg.update(dict(cfg.get("gpu_memory_monitor") or {}))
    if not enabled:
        return None

    allowed_keys = {
        "poll_interval_seconds",
        "sample_every_n_steps",
        "sync_cuda",
        "sample_on_batch_hooks",
        "backend",
        "log_prefix",
        "prog_bar",
    }
    kwargs = {key: value for key, value in monitor_cfg.items() if key in allowed_keys}
    return NvidiaSmiMemoryMonitor(enabled=True, **kwargs)
