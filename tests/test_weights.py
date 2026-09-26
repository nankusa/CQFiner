"""Standalone integrity checks; no dataset or CUDA required."""

import hashlib
import json
from pathlib import Path
import unittest
import torch
from src.config import load_training_config
from src.model import build_site_model

ROOT = Path(__file__).resolve().parents[1]


class ReleaseWeightsTest(unittest.TestCase):
    def test_main_configuration_and_exact_state_keys(self):
        config = load_training_config(ROOT / "configs/train.yaml")
        self.assertTrue(config.resolved["model"]["backbone"]["egnn"]["norm_feats"])
        self.assertEqual(config.resolved["model"]["query"]["num_nodes"], 200)
        model = build_site_model(config.train["model"])
        path = ROOT / "weights/cqfiner.pt"
        meta = json.loads((ROOT / "weights/metadata.json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), meta["sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(set(payload), {"state_dict", "epoch", "global_step", "format"})
        extra = {k for k in payload["state_dict"] if not k.startswith("model.")}
        self.assertEqual(extra, {"shared_query_embed"})
        self.assertEqual(
            tuple(payload["state_dict"]["shared_query_embed"].shape), (1, 1280)
        )
        state = {
            k.removeprefix("model."): v
            for k, v in payload["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state, strict=True)
        self.assertTrue(
            all(torch.isfinite(v).all() for v in payload["state_dict"].values())
        )


if __name__ == "__main__":
    unittest.main()
