# ✈️ Airport Regime — Regime-Adaptive Prediction of Near-Term Flight Disruptions

**Core question:** Does an airport operate under different latent conditions, and does recognizing those conditions improve near-term disruption prediction?

> This is an applied ML experiment, not a claim of a new algorithm.

## TL;DR

I built a pipeline that turns raw flight events into an hourly operational time series per airport, trained strong tabular models (LightGBM, CatBoost) to predict near-term disruption, then tested whether adding *unsupervised latent operating regimes* (via Gaussian Mixture Models) improved that prediction.

**Result: no.** Regime-awareness moved test-set PR-AUC by **-0.001** — noise-level. SHAP explains why: the regimes were built from the same signals (delay rate, inbound delay, momentum) already available to the model directly, so the cluster label added no new information. The strongest driver of near-term disruption risk is **inbound delay pressure from connecting flights**, not the discovered regime.

Negative results are still results — this repo reports the honest outcome rather than only publishing the version that "worked."

## Pipeline

```
flight events → airport-hour state → exact temporal lags → tabular ML → latent regimes → regime-aware ML → explainability
```

1. **Scope** — Start from ~3M U.S. flight records (2019–2023) and select the 8 busiest, most delay-prone airports (DEN, DFW, ORD, LAS, MCO, PHX, EWR, LAX), using training-period data only.
2. **Build an hourly panel** — Convert flight-level events into an airport × hour operational time series: scheduled/operated departures, delay rate, cancellation rate, and inbound pressure from connecting flights.
3. **Add temporal features** — Lags at 1h, 2h, 3h, 6h, 24h, and 168h (one week), rolling means/volatility, and momentum/acceleration of the delay rate.
4. **Define the target** — "Disruption" = the next 1–2 operating hours are unusually delayed relative to that airport's own historical top decile (threshold learned from training data only).
5. **Baseline models** — Seasonal empirical prior → Logistic Regression → LightGBM → CatBoost, selected on validation PR-AUC before the test set is ever touched.
6. **Discover latent regimes** — Fit a Gaussian Mixture Model (BIC-selected, 4 states) on operational features to see if airports cycle through hidden "modes" (calm, elevated, high-delay, rapidly-worsening).
7. **Regime-aware model** — Add the discovered regime as a feature to the winning booster; re-evaluate on the untouched test set.
8. **Explain it** — SHAP values on the final model to see what's actually driving predictions.

## Data

[Flight Delay and Cancellation Dataset 2019–2023](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023) (Kaggle, patrickzel), 3M-row sample. Not redistributed here — download it from the source and point `DATA_PATH` at your local copy to reproduce.

## Methodology notes (why this should be trusted)

- **Chronological split**, no shuffling: train ≤ 2022, validation = 2022, test = 2023 (final period never touched during model selection).
- **No leakage**: imputation statistics, the disruption threshold, and the regime clusters are all fit on training data only, then applied to validation/test.
- **Lag features are shifted correctly** (`shift(1)` before any rolling calculation) so no hour ever sees its own future.
- **Model selection happens on validation only** — the test set is used exactly once, for final reporting.

## Key result

| Model | PR-AUC (test) |
|---|---|
| Static CatBoost (best booster) | baseline |
| + Latent regime feature | Δ ≈ **-0.001** |

## What actually drives the prediction (SHAP)

Top features by mean absolute SHAP value:

1. Inbound delay state (connecting-flight pressure)
2. Average departure delay
3. Inbound delay lag (1h)
4. Current delay rate
5. Rolling 6h delay average

The discovered regime label does **not** rank among the top drivers — it's redundant with information the model already has.

## Reproducing

```bash
pip install pandas numpy scikit-learn lightgbm catboost shap plotly
```

Update `DATA_PATH` in the first cell to point at your local copy of the dataset, then run cells top to bottom.

## Literature context

This is an applied synthesis, not a new-method paper. Related recent work:

- Flight-delay ML systematic review (2026): https://doi.org/10.1016/j.asoc.2026.115761
- Regime-switching aviation forecasting (2026): https://doi.org/10.1016/j.jairtraman.2026.103055
- Tabular foundation-model benchmark (Nature): https://doi.org/10.1038/s41586-024-08328-6

## License

Code in this repo: MIT (or your choice). Dataset is third-party — see the Kaggle listing for its license before redistributing any derived data.
