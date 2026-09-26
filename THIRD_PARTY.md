# Third-party components

- The EGNN implementation follows [VN-EGNN](https://github.com/ml-jku/vnegnn).
  The original MIT notice is retained in `licenses/VN-EGNN.txt`.
- The geometric encoder follows [ViSNet](https://github.com/microsoft/ViSNet).
  Microsoft's [MIT notice](https://github.com/microsoft/ViSNet/blob/main/LICENSE)
  is retained in `licenses/ViSNet.txt`.
- Frozen residue embeddings use
  [ESM2](https://huggingface.co/facebook/esm2_t33_650M_UR50D); the ESM model itself
  is not included in `weights/cqfiner.pt`.
- Data and the AP evaluation protocol follow
  [UniSite](https://github.com/quanlin-wu/unisite). Dataset distribution and terms
  are separate from this code license; consult the upstream distribution.

PyTorch, PyTorch Geometric, Lightning, BioPython, SciPy and other installed
packages retain their own licenses. Installation references:
[PyTorch 2.6](https://pytorch.org/get-started/previous-versions/),
[PyG CUDA 12.4 wheels](https://data.pyg.org/whl/torch-2.6.0+cu124.html).
