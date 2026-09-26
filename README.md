# CQFiner

Protein binding site prediction with local directed residue–query graphs, a
ViSNet backbone, three EGNN refinement layers, and distance-gated residue masks.
This repository provides pretrained weights, training and evaluation code,
dataset splits, preprocessing tools, and reference results.

## Evaluate the pretrained model

The reference environment uses Linux, Python 3.12, and PyTorch 2.6.0 with CUDA
12.4. The evaluation commands below use one NVIDIA GPU with mixed precision
and are intended to be run from this directory.

```bash
bash scripts/install.sh
.venv/bin/python -B scripts/check_data.py --data-root /path/to/prepared/data
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B src/evaluate.py --config configs/eval.yaml \
  --data-root /path/to/prepared/data
.venv/bin/python -B scripts/check_results.py \
  --metrics lightning_logs/evaluation/main/test_metrics.csv
```

The supplied `weights/cqfiner.pt` was exported from the checkpoint selected at
epoch 17 using UniSite validation AP@0.5. It is approximately 9 MB,
loads with `torch.load(..., weights_only=True)`, and does not require Git LFS.
SHA-256 and provenance are recorded in [weights/metadata.json](weights/metadata.json).
It contains inference weights, not optimizer state for resuming the original run.

The table below reports results from the original experiments for reference.

| Dataset | Proteins | AP@0.3 | AP@0.5 | DCC@4 | DCA@4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| COACH420 | 287 | 0.815765 | 0.633765 | 0.735119 | 0.860119 |
| HOLO4K | 1,629 | 0.805384 | 0.639687 | 0.766776 | 0.877053 |

[Reference values](results/expected.csv) are also available in CSV format. AP
uses residue masks; DCC/DCA use direct query centers and top-*n* selection at 4 Å.
The separate `site_dcc`/`site_dca` fields describe mask-derived centers.

The supplied evaluation setup uses 200 queries, FPS sampling, batch size 8,
the provided split order, FP16, and EGNN feature normalization. We recommend
starting with these settings when comparing against the reference results.
Graph-level feature normalization makes predictions sensitive to batch
composition; hardware, software versions, and numerical precision may also
affect the metrics.

The initial standalone evaluation differs slightly from some reference values
and has not passed the strict numerical comparison. See the
[recorded metrics](results/measured.csv), [comparison report](results/verification.json),
and [verification notes](docs/reproduction.md) for the current results.

## Data

Prepared data are external to the code/weights package. See [docs/data.md](docs/data.md)
for the expected layout, dataset splits, preprocessing, and portable bundle
creation. Use `--data-root` to point to prepared data following that layout.

## Train

The provided training configuration uses two GPUs, batch size 8 per GPU,
effective batch size 16, seed 42, AdamW at 2e-4, 60 epochs, and 200 surface queries.

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B src/train.py --config configs/train.yaml \
  --data-root /path/to/prepared/data --logger-version seed42
```

TensorBoard events, resolved configurations and checkpoints are written under
`lightning_logs/cqfiner/seed42/`. The `best-val_site_ap_iou_0p5-*.ckpt` checkpoint
follows the selection criterion used in the original experiments. Evaluate a
newly trained model by passing `--checkpoint /path/to/best.ckpt` to
`src/evaluate.py`.

To resume training from a full Lightning checkpoint, pass `--ckpt-path` to
`src/train.py`. The provided compact weight file is for evaluation.

The validation entry point follows the two-GPU batching setup used for UniSite
validation during training:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B src/validate.py --data-root /path/to/prepared/data
```

The original training run recorded a validation AP@0.5 of 0.502081; this value
has not been recomputed as part of the release checks. Training from scratch
also introduces stochastic variation, so the pretrained weights provide a
useful starting point for evaluation.

## Verification tools

```bash
OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests
OMP_NUM_THREADS=1 .venv/bin/python -B scripts/check_training.py \
  --data-root /path/to/prepared/data
```

The second command checks the training pipeline with two Lightning optimizer
steps, validation, and checkpoint reloading, then removes temporary outputs.
See [docs/reproduction.md](docs/reproduction.md) for the scope of these checks
and the reference experimental settings.

## Layout

```text
configs/       main model, training and evaluation settings
src/           data, geometric networks, losses, Lightning and evaluation
scripts/       environment setup, data checks and result verification
weights/       selected main-model weights and SHA-256
results/       reference and standalone evaluation metrics
data/splits/   benchmark and training split lists
tests/         checkpoint integrity checks
docs/          data and reproduction instructions
licenses/      third-party notices
```

The internal `SurfQNet` class name is retained for compatibility; CQFiner is the
paper method name.

## License and attribution

Original project code is provided under the [MIT license](LICENSE). Notices for
adapted components are retained in [licenses/](licenses/) and
[THIRD_PARTY.md](THIRD_PARTY.md). Datasets and pretrained ESM assets retain their
upstream terms and are distributed separately.
