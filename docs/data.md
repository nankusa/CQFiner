# Data preparation

## Prepared data contract

Use the same normalized structures, site annotations, surface targets and ESM
features as the main experiment. The source datasets are described by
[UniSite](https://github.com/quanlin-wu/unisite/blob/main/DATASETS.md) and distributed
through its [official dataset page](https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1).
These upstream assets have their own terms; this code package does not change them.
The release's `data/splits/` lists define the evaluated subsets and their order.
Upstream archives are not automatically interchangeable with this prepared layout.

```text
prepared_data/
  coach420/                         # likewise holo4k and unisite
    protein_ligand/
      1a26A/
        protein.pdb                 # normalized protein coordinates
        ligand_site1.pdb            # one normalized ligand file per site
        pocket_site1.txt            # annotated residue IDs (benchmark sets)
        site_manifest.csv           # site, ligand_file, pocket_file mapping
    esm/
      1a26A.npy                     # [number of residues, 1280], float32
    pocket/
      1a26A.npz                     # prepared surface and site targets
    splits/
      test_ids_coach420             # exact ordered release split
      all_ids_coach420
```

UniSite also retains `source_info.csv` and `source.mapping` where present;
these carry binding-residue mappings used by the residue-mask loader. Residue
features must follow the protein parser's residue order. Surface queries use
cached solvent-accessible atom coordinates; graph edges are built online and
no `graph_residue` or `graph_atom` directories are required.

COACH420 contains 287 evaluated proteins and HOLO4K contains 1,629. UniSite
validation contains 2,286. Benchmark names refer to the original collections,
not the sizes after preprocessing. The exact lists, rather than the nominal
names, define this release. The split cleanup is recorded in
`data/splits/cleanup.json`.

Existing prepared datasets need no copying:

```bash
.venv/bin/python -B scripts/check_data.py --data-root /path/to/prepared_data
.venv/bin/python -B scripts/check_data.py --data-root /path/to/prepared_data \
  --datasets unisite --partition valid
```

The checker rejects missing files and any change in split membership or order.
Tensor-level shape and finiteness checks run in the dataset loader.

## Regenerate features for normalized structures

Install the optional ESM preprocessing dependencies and obtain the exact ESM2
model directory. The extractor requires a resolved local model directory; it
does not select an arbitrary cache revision or fall back from CUDA to CPU.

```bash
uv pip install --python .venv/bin/python -r requirements-esm.txt
.venv/bin/hf download facebook/esm2_t33_650M_UR50D --local-dir weights/esm2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m src.data.build_esm_embeddings \
  --path /path/to/prepared_data/coach420 --model-path weights/esm2 \
  --device cuda --n-jobs 16
```

Repeat ESM extraction for each required dataset. Copy the corresponding release
split files from `data/splits/<dataset>/` into `<dataset>/splits/`. The normalized
raw files and ESM features must cover their `all_ids` lists before building
pocket targets:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -B -m src.data.build_pockets \
  --data-root /path/to/prepared_data --datasets coach420 holo4k --workers 16
```

Add `unisite` when preparing training data. Each worker uses one compute thread.
Existing pocket output directories are never overwritten. UniSite surface-site
labels use a 4 Å ligand-distance threshold; benchmark sets use supplied residue
annotations. Surface atom selection uses SASA > 1e-4 without MSMS residue depths.
Changing structures, labels, surface points or ESM extraction can change the
reported metrics; reuse the prepared features for exact checkpoint evaluation.

## Export a portable data bundle

For maintainers with the exact prepared data, this command creates a test-data
archive containing the required structures, labels, embeddings and pocket files:

```bash
.venv/bin/python -B scripts/package_data.py --data-root /path/to/prepared_data \
  --datasets coach420 holo4k --partition test --output /path/to/cqfiner_test_data.tar
```

Extract it into a fresh prepared-data directory, then run `check_data.py`.
The archive excludes private `sample_manifest.json` files, unrelated modalities,
absolute source paths and the research repository. Required mapping sidecars
are preserved. For training, use `--datasets unisite --partition all` and also
copy the released `train_ids_unisite_0p9` and `valid_ids_unisite_0p9` lists into the
extracted UniSite `splits/` directory.

The code/weights release does not embed the multi-gigabyte prepared datasets or
pretend that a data download has been published. Prepared bundles can be hosted
separately under the applicable dataset terms.
