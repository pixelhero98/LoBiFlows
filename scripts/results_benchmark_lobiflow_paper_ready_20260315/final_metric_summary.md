# Final Metric Summary

_Generated from the published paper-ready metric catalogs; do not hand-edit._

This file summarizes the accepted paper-ready LoBiFlow results. Lower is better
for all metrics except `TSTR MacroF1`, where higher is better.

## Accepted Quality Presets

- `synthetic`: `transformer`, `history_len=128`, `solver=euler`, `eval_nfe=2`
- `optiver`: `transformer`, `history_len=128`, `solver=dpmpp2m`, `eval_nfe=4`
- `cryptos`: `hybrid`, `history_len=256`, `solver=dpmpp2m`, `eval_nfe=2`
- `es_mbp_10`: `hybrid`, `history_len=256`, `solver=euler`, `eval_nfe=1`

## LoBiFlow Quality Results

All numbers are `mean +/- std` over `5` seeds.

| Dataset | `score_main` | `TSTR MacroF1` | `Disc.AUC Gap` | `U-W1` | `C-W1` | `efficiency_ms_per_sample` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `synthetic` | `0.3711 +/- 0.0105` | `0.4270 +/- 0.0163` | `0.3994 +/- 0.0165` | `0.2501 +/- 0.0101` | `0.3358 +/- 0.0162` | `2547.5 +/- 61.5` |
| `optiver` | `0.3595 +/- 0.0118` | `0.3019 +/- 0.0276` | `0.1422 +/- 0.0184` | `0.2082 +/- 0.0133` | `0.5076 +/- 0.0178` | `769.6 +/- 1.6` |
| `cryptos` | `1.8300 +/- 0.1070` | `0.1596 +/- 0.0122` | `0.4999 +/- 0.0001` | `63.0028 +/- 25.9818` | `60.9227 +/- 25.4083` | `5245.4 +/- 20.8` |
| `es_mbp_10` | `0.4569 +/- 0.0105` | `0.3377 +/- 0.0233` | `0.4945 +/- 0.0063` | `0.3764 +/- 0.0145` | `0.4223 +/- 0.0260` | `2516.3 +/- 43.6` |

## Main Comparison Against Baselines

Best baseline on `score_main` for each dataset:

| Dataset | LoBiFlow | Best baseline | Outcome |
| --- | ---: | ---: | --- |
| `synthetic` | `0.3711 +/- 0.0105` | `TimeGAN 0.5107 +/- 0.0034` | LoBiFlow wins |
| `optiver` | `0.3595 +/- 0.0118` | `KoVAE 0.4045 +/- 0.0110` | LoBiFlow wins |
| `cryptos` | `1.8300 +/- 0.1070` | `CGAN 2.1919 +/- 0.3634` | LoBiFlow wins |
| `es_mbp_10` | `0.4569 +/- 0.0105` | `TimeCausalVAE 0.5507 +/- 0.0484` | LoBiFlow wins |

## Full 4+7 Metric Tables

The full `4` primary + `7` diagnostic metrics are stored in:

- `scripts/results_benchmark_lobiflow_paper_ready_20260315/metric_catalog.csv`
- `scripts/results_model_metric_catalogs_20260316/all_models_metric_catalog.csv`

These CSVs are the main comparison entry points for:

- LoBiFlow quality variants
- LoBiFlow speed variants
- LoBiFlow architecture ablations
- all five external baselines
