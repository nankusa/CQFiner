# CQFiner

Protein binding site prediction with local directed residue–query graphs, a
ViSNet backbone, three EGNN refinement layers, and distance-gated residue masks.
This release includes the **exact selected main-model weights**, training and
evaluation code, canonical splits, preprocessing tools, and reference results.

## Reproduce the main test results

Linux, Python 3.12, PyTorch 2.6.0 with CUDA 12.4, and one NVIDIA GPU are required
for the published mixed-precision evaluation. Run commands from this directory.

```bash
bash scripts/install.sh
.venv/bin/python -B scripts/check_data.py --data-root /path/to/prepared/data
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B src/evaluate.py --config configs/eval.yaml \
  --data-root /path/to/prepared/data
.venv/bin/python -B scripts/check_results.py \
  --metrics lightning_logs/evaluation/main/test_metrics.csv
```

The supplied `weights/cqfiner.pt` contains all learned tensors from the checkpoint
selected at **epoch 17 by UniSite validation AP@0.5**. It is approximately 9 MB,
loads with `torch.load(..., weights_only=True)`, and does not require Git LFS.
SHA-256 and provenance are recorded in [weights/metadata.json](weights/metadata.json).
It contains inference weights, not optimizer state for resuming the original run.

| Dataset | Proteins | AP@0.3 | AP@0.5 | DCC@4 | DCA@4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| COACH420 | 287 | 0.815765 | 0.633765 | 0.735119 | 0.860119 |
| HOLO4K | 1,629 | 0.805384 | 0.639687 | 0.766776 | 0.877053 |

[Reference values](results/expected.csv) retain full precision. AP uses residue
masks; DCC/DCA use direct query centers and top-*n* selection at 4 Å. The separate
`site_dcc`/`site_dca` fields describe mask-derived centers and are not the paper's
main localization metrics.

**Reproduction settings matter:** keep 200 queries, FPS sampling, batch size 8,
the supplied split order, FP16, and **EGNN feature normalization enabled**.
Graph-level feature normalization makes batch composition relevant. Changing
these settings can change predictions even with identical weights. Evaluation
raises on missing files, invalid samples, checkpoint mismatches, and nonfinite
outputs; it does not substitute another model or skip failed proteins.

## Data

Prepared data are external to the code/weights package. See [docs/data.md](docs/data.md)
for the exact layout, canonical splits, preprocessing, and portable bundle
creation. Existing prepared data can be used directly via `--data-root`; no
symlink or access to a parent research repository is required.

## Train

The main configuration uses two GPUs, batch size 8 per GPU, effective batch size
16, seed 42, AdamW at 2e-4, 60 epochs, and 200 surface queries.

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B src/train.py --config configs/train.yaml \
  --data-root /path/to/prepared/data --logger-version seed42
```

TensorBoard events, resolved configurations and checkpoints are written under
`lightning_logs/cqfiner/seed42/`. Select `best-val_site_ap_iou_0p5-*.ckpt` for the
paper's checkpoint-selection rule. Evaluate a newly trained model by passing
`--checkpoint /path/to/best.ckpt` to `src/evaluate.py`.

To resume **your own full Lightning checkpoint**, pass `--ckpt-path` to
`src/train.py`. The provided compact weight file is for evaluation.

UniSite validation used two-GPU rank-strided batches in training. Preserve that
configuration when re-evaluating the selected weights:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B src/validate.py --data-root /path/to/prepared/data
```

The original recorded validation AP@0.5 is 0.502081. Small floating-point
variations may occur across hardware. Training from scratch is stochastic;
reproducing inference from the released checkpoint is the primary verification.

## Check the release

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests
OMP_NUM_THREADS=1 .venv/bin/python -B scripts/check_training.py \
  --data-root /path/to/prepared/data
```

The second command runs two Lightning optimizer steps with the full architecture,
validates, checks checkpoint reloading, and deletes temporary outputs. It is a
pipeline check, not a new scientific result. See [docs/reproduction.md](docs/reproduction.md)
for release verification and exact experimental settings.

## Layout

```text
configs/       main model, training and evaluation settings
src/           data, geometric networks, losses, Lightning and evaluation
scripts/       environment setup, data checks and result verification
weights/       selected main-model weights and SHA-256
results/       expected and reproduced test metrics
data/splits/  exact ordered benchmark and training split lists
tests/         checkpoint integrity checks
docs/          data and reproduction instructions
licenses/      third-party notices
```

The internal `SurfQNet` class name is retained for compatibility; CQFiner is the
paper method name. No absolute workstation paths are required by the release.

## License and attribution

Original project code is provided under the [MIT license](LICENSE). Notices for
adapted components are retained in [licenses/](licenses/) and
[THIRD_PARTY.md](THIRD_PARTY.md). Datasets and pretrained ESM assets retain their
upstream terms and are distributed separately.
