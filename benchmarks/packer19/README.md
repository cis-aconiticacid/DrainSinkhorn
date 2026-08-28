# Packer19 endpoint

This benchmark builds a real temporal EOT workload from the Packer et al. 2019
*C. elegans* embryogenesis single-cell atlas. Every candidate maps one observed
cell support to the next and feeds the verified transport plan into a
matrix-free cell-type transition consumer.

## Data

Download `GSE126954.h5ad` from
[Zenodo 7496490](https://doi.org/10.5281/zenodo.7496490). The preparation script
checks the published MD5, never duplicates observations, selects cells by a
stable hash, and writes array hashes to a sidecar metadata file.

## Prepare

```bash
pip install -e ".[packer19]"
python benchmarks/packer19/prepare.py \
  --input /data/GSE126954.h5ad \
  --output benchmark_outputs/packer19.npz \
  --expected-md5 bcc0c311826766d7f9de1d6c01856027 \
  --support-mode sliding --n 16384 --width 8 --stride 4096 \
  --world-size 4 --components 64
```

## Run

```bash
python benchmarks/packer19/run_modes.py \
  --data benchmark_outputs/packer19.npz \
  --output benchmark_outputs/packer19_modes.json \
  --width 8 --repeats 3
```

The four public modes are upstream Flash serial, fixed-width candidate packing,
logical masking, and physical compaction. Use the logical-mask/physical-
compaction ratio to isolate eliminated physical work.

## Analyze the formal two-host roots

```bash
python benchmarks/packer19/analyze.py \
  --run-root /evidence/host0/raw_formal \
  --run-root /evidence/host1/raw_formal \
  --output benchmark_outputs/c80_lm_analysis
```

The analyzer intentionally refuses incomplete roots, repeated host identity,
configuration drift, source drift, input drift, release-semantic mismatch, or
missing telemetry.
