# Structured Conditional Regularization Ablation

This study summarizes follow-up regularizers added on top of the final LoBiFlow architecture. "Structured conditional regularization" refers to losses that encode explicit assumptions about conditional trajectory structure, irreversibility, or history-local future coupling.

## Evaluated Regularizers

| Regularizer | Main result | Status |
| --- | --- | --- |
| History-local causal OT | improves `cryptos` across the fresh 20k sweep; neutral-to-negative elsewhere | dataset-specific candidate |
| Global causal OT | helps `cryptos`, hurts `optiver` | not promoted to default |
| Conditional current matching (raw) | hard regression | rejected |
| Conditional current matching (Huber + shrink + selected currents) | improves `cryptos` across the fresh 20k sweep; negative on the other three datasets by 20k | dataset-specific candidate |
| MI (InfoNCE) | negative on `optiver` and `cryptos` | rejected |
| MI with frozen critic | negative on `optiver` | rejected |
| Path-space conditional FM approximation | negative on `optiver` and `cryptos` | rejected |

## High-Level Findings

1. The final paper-ready LoBiFlow defaults remain the robust open-source default.
2. Structured conditional regularizers are not universal improvements.
3. A fresh 3-seed, 20k-step sweep shows both history-local causal OT and conditional current matching help `cryptos`, but `synthetic`, `optiver`, and `es_mbp_10` are neutral-to-negative by the final checkpoint.
4. The diagnostics are still useful: they identify when a dataset has the local future structure needed for the regularizers to help.

## Pilot Visualizations

These figures summarize both applicability and training benefit/drawback:

1. local causal OT helps when local future laws are concentrated and stable
2. current matching helps when a large share of path current is locally predictable
3. the new 20k curves compare training effects across all four datasets
4. the 2x2 PDF combines both applicability plots and both all-dataset training-delta plots

![History-local causal OT applicability](causal_ot_applicability.png)

![Conditional current matching applicability](current_matching_applicability.png)

![Causal OT checkpoint curve on cryptos](causal_ot_checkpoint_curve_cryptos.png)

![Conditional current matching checkpoint curve on cryptos](current_matching_checkpoint_curve_cryptos.png)

![20k regularization training deltas](regularization_training_delta_20k.png)

Publication PDF: `structured_regularization_ablation_2x2.pdf`.

## Fresh 20k Cross-Dataset Sweep

The fresh sweep was run from scratch with `max_steps=20000`, not appended from the older 12k jobs. It uses datasets `cryptos`, `synthetic`, `optiver`, and `es_mbp_10`; variants `baseline_fm`, `local_causal_ot`, and `conditional_current_matching`; seeds `0,1,2`; checkpoints `1000,2000,4000,8000,12000,16000,20000`; horizons `60/300/900`; and `20` evaluation windows.

Final `20k` score_main values:

| Dataset | Baseline | Local causal OT | Delta | Current matching | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cryptos` | `1.9635` | `1.6801` | `-0.2834` | `1.6499` | `-0.3136` |
| `synthetic` | `0.3785` | `0.3891` | `+0.0106` | `0.4049` | `+0.0264` |
| `optiver` | `0.3548` | `0.3655` | `+0.0107` | `0.3626` | `+0.0078` |
| `es_mbp_10` | `0.4691` | `0.4872` | `+0.0181` | `0.4955` | `+0.0264` |

Lower `score_main` is better. Negative delta means the regularizer improved over the baseline. The result is clear: both regularizers are useful on `cryptos`, but neither is a generic default across the other datasets.

## Causal OT

### Short-Budget Result

`cryptos`, `3` seeds, `4000` steps, local causal OT with `lambda=0.01`, `horizon=4`, `k_neighbors=8`, `history_weight=0.1`:

| Variant | `score_main` | `TSTR` | `U-W1` | `C-W1` |
| --- | ---: | ---: | ---: | ---: |
| baseline | `1.744 +/- 0.161` | `0.205 +/- 0.058` | `16.859 +/- 5.522` | `16.650 +/- 5.386` |
| local causal OT | `1.565 +/- 0.113` | `0.231 +/- 0.040` | `11.336 +/- 2.670` | `11.318 +/- 2.645` |

This was the strongest positive result among the original short-budget structured regularizers.

### Applicability Diagnostics

The useful diagnostics were:

- `local_global_dispersion_ratio`: how much the local future law contracts relative to the global future law
- `neighborhood_stability_k_vs_2k`: how sensitive the local target is to neighborhood size

Lower is better for both.

| Dataset | Local/global dispersion | Neighbor stability | Observed effect |
| --- | ---: | ---: | --- |
| `synthetic` | `0.0172` | `0.0095` | negative |
| `optiver` | `0.4461` | `0.0203` | negative |
| `cryptos` | `0.0228` | `0.0059` | positive |
| `es_mbp_10` | `0.4052` | `0.0182` | weak/mixed |

Interpretation: causal OT helps when local histories sharply narrow the future-path distribution and those neighborhoods are stable. `cryptos` satisfies both conditions best; `synthetic` also has locality, but the baseline leaves less useful headroom for this OT signal.

### Training-Stage Dependence

`cryptos`, `3` seeds, checkpoints `1k/2k/4k/8k/12k/16k/20k`, macro over horizons `60/300/900`, `20` evaluation windows:

| Steps | Baseline `score_main` | Local OT `score_main` | Delta |
| --- | ---: | ---: | ---: |
| `1k` | `2.6924` | `2.6504` | `-0.0420` |
| `2k` | `1.9909` | `1.7640` | `-0.2269` |
| `4k` | `1.9881` | `1.7797` | `-0.2084` |
| `8k` | `1.9945` | `1.8878` | `-0.1067` |
| `12k` | `1.6945` | `1.5270` | `-0.1675` |
| `16k` | `1.8727` | `1.6669` | `-0.2058` |
| `20k` | `1.9635` | `1.6801` | `-0.2834` |

The 20k rerun changes the earlier 12k-only reading: local causal OT remains positive on `cryptos` through `20k`, but the cross-dataset sweep shows this is not a robust default behavior.

### Full-Budget Crypto Refresh

The earlier stronger local causal OT configuration did not replace the accepted final crypto preset.

| Variant | `score_main` | `TSTR` | `U-W1` | `C-W1` |
| --- | ---: | ---: | ---: | ---: |
| baseline | `1.8150 +/- 0.0571` | `0.1467 +/- 0.0013` | `65.69 +/- 18.36` | `62.45 +/- 15.83` |
| local causal OT, `h=8` | `1.8455 +/- 0.0445` | `0.1842 +/- 0.0259` | `90.83 +/- 4.62` | `90.36 +/- 4.23` |

This improved `TSTR` but degraded the main score and both Wasserstein metrics.

## Conditional Current Matching

### Raw Formulation

The raw squared-error formulation on the full antisymmetric current vector was not usable. It produced large regressions on `cryptos`.

### Safer Formulation

Changes:

- Huber loss on normalized residuals
- local target shrunk toward a global mean
- selected current components only

`cryptos`, `3` seeds, `4000` steps:

| Variant | `score_main` | `TSTR` | `U-W1` | `C-W1` |
| --- | ---: | ---: | ---: | ---: |
| baseline | `1.7361` | `0.2105` | `16.5091` | `16.3398` |
| `lambda=0.0005` | `1.5006` | `0.2143` | `9.8496` | `9.8338` |
| `lambda=0.001` | `1.6065` | `0.1953` | `13.2739` | `13.2781` |

So the safer version gave a real short-budget gain on `cryptos`.

### Training-Stage Dependence

`cryptos`, `3` seeds, checkpoints `1k/2k/4k/8k/12k/16k/20k`, macro over horizons `60/300/900`, `20` evaluation windows:

| Steps | Baseline `score_main` | Current matching `score_main` | Delta | Current `TSTR` n |
| --- | ---: | ---: | ---: | ---: |
| `1k` | `2.6924` | `2.6098` | `-0.0826` | `3` |
| `2k` | `1.9909` | `1.8130` | `-0.1779` | `3` |
| `4k` | `1.9881` | `1.8074` | `-0.1807` | `3` |
| `8k` | `1.9945` | `1.8690` | `-0.1255` | `2` |
| `12k` | `1.6945` | `1.6355` | `-0.0590` | `3` |
| `16k` | `1.8727` | `1.6254` | `-0.2473` | `3` |
| `20k` | `1.9635` | `1.6499` | `-0.3136` | `3` |

The fresh 20k sweep shows conditional current matching is useful on `cryptos`, with the strongest mean improvement at `20k`. The same run also shows why it should not be promoted as a default: it degrades all three other datasets at the final checkpoint.

### Applicability Diagnostics

Useful diagnostics:

- `predictable_current_share = local_mean_norm / current_norm`
- `neighborhood_stability_k_vs_2k`

| Dataset | Predictable current share | Neighbor stability | Observed effect |
| --- | ---: | ---: | --- |
| `synthetic` | `0.49` | `1.015` | negative |
| `optiver` | `0.61` | `1.249` | negative |
| `cryptos` | `0.79` | `1.084` | positive |
| `es_mbp_10` | `0.44` | `1.066` | negative |

Interpretation: current matching helps only when a large share of the path-antisymmetric current is history-local and predictable. `cryptos` was the only dataset that met that condition strongly enough.

### Full-Budget Crypto Refresh

The earlier paper-ready crypto refresh was negative for the main distribution metrics:

| Variant | `score_main` | `TSTR` | `U-W1` | `C-W1` |
| --- | ---: | ---: | ---: | ---: |
| baseline | `1.8300` | `0.1596` | `63.0028` | `60.9227` |
| safer current matching | `1.8942` | `0.1755` | `69.3745` | `69.1723` |

This improved `TSTR` but worsened the main score and both primary distribution-matching metrics.

## MI and Path-Space FM

These directions were negative in the tested forms:

- `MI` with in-batch InfoNCE: negative on `optiver` and `cryptos`
- `MI` with frozen history-future critic: negative on `optiver`
- path-space conditional FM approximation: negative on `optiver` and `cryptos`

The main issue was objective mismatch: the additional path-level signal was not strong or well-aligned enough to improve the final conditional generation metrics.

## Final Takeaway

The follow-up study supports a narrow conclusion:

- structured conditional regularizers can help in dataset-specific regimes
- neither history-local causal OT nor conditional current matching is a generic default
- `cryptos` is the positive case for both regularizers
- `synthetic`, `optiver`, and `es_mbp_10` show the training drawbacks that block default promotion
