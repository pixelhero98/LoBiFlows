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

## Usage

Run the main LoBiFlow suite with dataset-specific defaults:

```bash
cd Model
python experiments_icassp.py --dataset synthetic --out_dir results_synth
python experiments_icassp.py --dataset optiver --out_dir results_optiver
python experiments_icassp.py --dataset cryptos --out_dir results_cryptos
python experiments_icassp.py --dataset es_mbp_10 --out_dir results_es
```

Run the faster `NFE=1` speed variant:

```bash
cd Model
python experiments_icassp.py --dataset cryptos --lobiflow_variant speed --out_dir results_cryptos_speed
```

Run the paper-ready benchmark bundle:

```bash
cd Model
python benchmark_lobiflow_paper_ready.py
```

Export flat CSV/JSON metric catalogs:

```bash
cd Model
python export_model_metric_catalogs.py
```

## Hyperparameter Tuning

LoBiFlow applies dataset presets first, then CLI overrides. The main knobs are:

- data: `--dataset`, `--data_path`, `--synthetic_length`
- optimization: `--steps`, `--batch_size`, `--lr`, `--weight_decay`
- context: `--history_len`, `--ctx_encoder`, `--ctx_causal`, `--ctx_local_kernel`, `--ctx_pool_scales`
- sampling: `--eval_nfe`, `--solver`, `--lobiflow_variant`
- evaluation: `--eval_horizon`, `--rollout_horizons`, `--eval_windows_*`

Typical examples:

```bash
cd Model
python experiments_icassp.py --dataset cryptos --history_len 384 --ctx_encoder hybrid --ctx_local_kernel 7 --ctx_pool_scales 8,32
python experiments_icassp.py --dataset optiver --eval_nfe 4 --solver dpmpp2m
python experiments_icassp.py --dataset synthetic --synthetic_length 5000000 --steps 20000
```

Current quality presets:

- `synthetic`: `transformer`, `history_len=128`, `solver=euler`, `eval_nfe=2`
- `optiver`: `transformer`, `history_len=128`, `solver=dpmpp2m`, `eval_nfe=4`
- `cryptos`: `hybrid`, `history_len=256`, `solver=dpmpp2m`, `eval_nfe=2`
- `es_mbp_10`: `hybrid`, `history_len=256`, `solver=euler`, `eval_nfe=1`

## Final Outputs

Paper-ready benchmark outputs are written under:

- `Model/results_benchmark_lobiflow_paper_ready_20260315`
- `Model/results_model_metric_catalogs_20260316`

The flat CSVs in `results_model_metric_catalogs_20260316` are the easiest entry
point for comparing LoBiFlow against all baselines.
