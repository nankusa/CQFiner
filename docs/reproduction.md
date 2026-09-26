# Reproduction protocol and verification

## Exact main checkpoint

The included weights were selected at epoch 17 (zero-based epoch 16), global
step 9,775, by UniSite validation AP@0.5. All 230 state tensors, including the
shared initial query embedding, are preserved exactly. Optimizer states,
callbacks, local paths and Lightning loop state were removed from the public
weight file. Check `weights/metadata.json` before evaluation.

## Configuration

| Setting | Value |
| --- | --- |
| Protein nodes / features | residue Cα / frozen ESM2, 1,280 channels |
| Training / evaluation queries | 200 / 200 |
| Coarse encoder | 6-layer ViSNet, 128 channels |
| Graph cutoff / neighbor cap | 10 Å / 256 |
| Query refiner | 3-layer EGNN, 10 Å, 32 neighbors |
| EGNN feature normalization | enabled, as in the selected main experiment |
| Message direction | residue→query; no outgoing query edges |
| Mask head | adaptive distance-gated dot product |
| Query-center NMS / mask grouping | 6 Å / 7 Å |
| Seed / mixed precision | 42 / FP16 |
| Evaluation batch size | 8, one GPU, original split order |
| Training batch size | 8 per GPU × 2 GPUs |
| Optimizer / schedule | AdamW, 2e-4, plateau scheduler, 60 epochs |

The refiner builds its neighborhood once at coarse query positions, then updates
coordinates and distances while retaining those neighbor indices. Protein
features stay fixed within refinement. The main inference path uses one ViSNet
pass followed by three EGNN layers.

Do not substitute the Q32 scaling model or a later configuration with EGNN
feature normalization disabled. Neither represents the main table checkpoint.
The raw `site_dcc` metrics use a separate mask-derived centroid pipeline;
`query_dcc_topn` and `query_dca_topn` are the main paper's localization metrics.

## Checks completed during packaging

- Exact equality of every exported weight tensor to the original checkpoint.
- Strict loading into the standalone architecture, including the learned query embedding.
- An 8-protein real-data CPU comparison with the experiment's frozen source:
  all 16 recorded output tensors are exactly equal (maximum difference zero).
- Two real-data Lightning optimizer steps with the full model, one validation
  batch, and strict checkpoint reload. Temporary training files were removed.
- Clean project-local virtual environment installed with `uv`; core dependency
  versions are pinned in the requirements files.
- Full evaluation of 287 COACH420 and 1,629 HOLO4K proteins completed.

The initial full GPU evaluation has a small discrepancy from the historical
reference and has **not passed the strict numerical reproduction check**.
The supplied `results/expected.csv` preserves the historical values unchanged;
`results/measured.csv` records the initial standalone evaluation. The discrepancy
is tracked in `results/verification.json`. Further GPU diagnostics are pending
author confirmation. Do not describe this state as exact GPU reproduction.

UniSite validation metrics in the README are the original training-run values.
They have not been recomputed in this release check. `src/validate.py` retains
the two-GPU validation setup because normalization depends on batch composition.

## Interpretation of tests

`tests/test_weights.py` validates the public checkpoint format, checksum and
state keys without accessing datasets. `scripts/check_training.py` verifies the
training pipeline using two real examples. These checks establish that the
implementation and inputs can execute; they do not substitute for full benchmark
verification. `scripts/check_results.py` compares every main test metric against
the published values and fails on differences exceeding its stated tolerance.
