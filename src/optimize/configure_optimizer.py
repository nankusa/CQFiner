import fnmatch
from typing import Iterable

import torch


def _make_param_groups(named_parameters: Iterable, param_groups_cfg):
    named = [(n, p) for n, p in named_parameters if p.requires_grad]
    consumed = set()
    groups = []

    for cfg in param_groups_cfg:
        pats = cfg.get("patterns", [])
        params = []
        for name, p in named:
            if name in consumed:
                continue
            if any(fnmatch.fnmatch(name, pat) for pat in pats):
                params.append(p)
                consumed.add(name)
        if params:
            group = {k: v for k, v in cfg.items() if k != "patterns"}
            group["params"] = params
            groups.append(group)

    remaining = [p for n, p in named if n not in consumed]
    if remaining:
        groups.append({"params": remaining})
    return groups


def build_optimizer(module, cfg: dict):
    opt_name = cfg.get("name", "adamw").lower()
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.0)

    if cfg.get("param_groups"):
        params = _make_param_groups(module.named_parameters(), cfg["param_groups"])
    else:
        params = filter(lambda p: p.requires_grad, module.parameters())

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=cfg.get("betas", (0.9, 0.95)),
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=cfg.get("momentum", 0.9), weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    sch_cfg = cfg.get("scheduler")
    if not sch_cfg:
        return {"optimizer": optimizer}

    sch_name = sch_cfg.get("name", "reduce_on_plateau").lower()
    if sch_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch_cfg.get("mode", "min"),
            factor=sch_cfg.get("factor", 0.1),
            patience=sch_cfg.get("patience", 8),
            min_lr=sch_cfg.get("min_lr", 1e-6),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": sch_cfg.get("monitor", "val/loss"),
                "interval": "epoch",
                "frequency": 1,
            },
        }

    if sch_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.get("t_max", 200),
            eta_min=sch_cfg.get("eta_min", 1e-6),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    raise ValueError(f"Unsupported scheduler: {sch_name}")
