# Reproducibility

## Evidence-locked source

The public core was reconstructed from the complete source archive embedded in
the C80-LM formal run, commit
`d264c7d6a5324d6bfec394364141977fef144383`. The source archive was carried
inside the immutable host package rather than inferred from the later research
worktree.

The formal environment used:

- NVIDIA A100-SXM4-80GB;
- PyTorch 2.5.1+cu124;
- CUDA 12.4.1;
- Triton 3.1.0;
- FlashSinkhorn 0.3.3.post1;
- FlashSinkhorn tree SHA-256
  `1d2e2fa3bf7b76b80d760abeb25ee44a689bb2ca87522d1d783e2477acd21efd`.

See [`SOURCE_PROVENANCE.json`](../SOURCE_PROVENANCE.json) for the archived file
hashes and the namespace-only changes made for this public package.

## Public Packer19 reproduction

Download `GSE126954.h5ad` from
[Zenodo record 7496490](https://doi.org/10.5281/zenodo.7496490). The published
file MD5 is `bcc0c311826766d7f9de1d6c01856027`.

```bash
pip install -e ".[packer19]"

python benchmarks/packer19/prepare.py \
  --input /data/GSE126954.h5ad \
  --output benchmark_outputs/packer19.npz \
  --expected-md5 bcc0c311826766d7f9de1d6c01856027 \
  --support-mode sliding \
  --n 16384 --width 8 --stride 4096 \
  --world-size 4 --components 64

python benchmarks/packer19/run_modes.py \
  --data benchmark_outputs/packer19.npz \
  --output benchmark_outputs/packer19_modes.json \
  --width 8 --repeats 3
```

The public runner rotates method order and excludes one warm-up per mode. It
reports endpoint time, release residual, candidate depth, physical and logical
slots, peak allocation, and a matrix-free cell-transition consumer hash.

## Formal C80-LM analyzer

The checked-in `benchmarks/packer19/analyze.py` is the fail-closed analyzer used
for the paper. It requires both immutable four-GPU host roots, verifies their
portable manifests, proves disjoint physical host identity, checks data/source/
backend/configuration equality, validates per-lane release semantics, and only
then produces the paired host-stratified comparison.

```bash
python benchmarks/packer19/analyze.py \
  --run-root /evidence/host0/raw_formal \
  --run-root /evidence/host1/raw_formal \
  --output benchmark_outputs/c80_lm_analysis
```

The immutable raw archives are kept outside Git because they contain the
prepared 16,384-cell supports and full telemetry. Their package hashes are
registered in `SOURCE_PROVENANCE.json`.
