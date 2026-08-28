# Results

This directory contains compact, machine-readable outputs that support the
paper tables. Raw GPU roots are not committed because they include prepared
real-data supports and full telemetry.

`c80_lm/` is copied from analysis commit
`793c08cd5bff13b7f854aba43864bf0065de738c` without changing the numeric
payload. Its analyzer is available at `benchmarks/packer19/analyze.py`.

`paper_results.json` is the public claim index. It separates matched component
effects, complete-deployment comparisons, controlled interventions, and
application-level results.
