# LoBiFlow

LoBiFlow is a history-conditioned flow-matching model for generating level-2
limit order books. This project includes the training code, benchmark scripts,
dataset preparation utilities, and final experiment artifacts used for the
paper-ready evaluation.

## Scope

- model: conditional L2 LOB generation in parameterized book space
- objective: flow matching with minibatch optimal transport matching
- conditioning: transformer or hybrid history encoder
- evaluation: 4 primary metrics + 7 diagnostic metrics

## Datasets

- LOBSTER-calibrated synthetic data
- Optiver Realized Volatility Prediction
- Binance crypto LOB snapshots from Tardis
- Databento ES futures MBP-10

## Main Scripts

- `Model/experiments_icassp.py`: main LoBiFlow runner
- `Model/benchmark_lobiflow_paper_ready.py`: final quality / speed / architecture benchmark
- `Model/export_model_metric_catalogs.py`: flat metric catalog export
- `Model/test_lobiflow.py`: smoke and regression suite

## Final Outputs

Paper-ready benchmark outputs are written under:

- `Model/results_benchmark_lobiflow_paper_ready_20260315`
- `Model/results_model_metric_catalogs_20260316`

The flat CSVs in `results_model_metric_catalogs_20260316` are the easiest entry
point for comparing LoBiFlow against all baselines.
