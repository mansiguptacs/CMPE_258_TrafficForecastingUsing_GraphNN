Traffic-Speed Forecasting Using Graph Neural Networks
Deep Learning Course 258-03, Fall 2026, San José State University


----------

Team Members: Lavya Sri Bonagiri(017552834), Mansi Gupta (020088497), Megha Gangal (020096869)


----------

Links:

- GitHub repository: https://github.com/mansiguptacs/CMPE_258_TrafficForecastingUsing_GraphNN
- Google Drive folder (dataset, processed data, reports): https://drive.google.com/drive/folders/1zijctqUvd5rywZyIUK8cRHI3JtknPYhq?usp=sharing
----------


Accurate traffic prediction can help drivers choose better routes, reduce traffic congestion and improve traffic-light control. However, many traditional time-series models only study how traffic changes over time and do not consider how nearby roads affect one another.
This project predicts traffic speed at each sensor 15, 30 and 60 minutes into the future, corresponding to 3, 6 and 12 steps ahead, respectively, because the dataset uses a five-minute sampling interval. The project represents the traffic sensor network as a graph, where each sensor is a node and edges encode road connectivity and the distance between sensors. This structure allows the model to learn both how traffic changes over time and how conditions at one sensor may affect connected sensors.


----------

Repository layout:

| Path | Contents |
|------|----------|
| `configs/metr_la.yaml` | Shared settings for windows, horizons, splits, normalization, graphs, and STGCN hyperparameters |
| `src/data/` | Data loading, chronological splitting, normalization, missing-value handling, sliding windows, and training-time input masking (`augment.py`) |
| `src/eval/` | MAE, RMSE and MAPE (`metrics.py`), the shared missing-data test for any model (`masked_inputs.py`), the STGCN missing-data evaluation (`missing_eval.py`), and the cross-model comparison (`compare.py`) |
| `src/models/stgcn/` | STGCN model, graph construction, training, and the experiment and hyperparameter-search queue |
| `notebooks/01_data_preprocess.ipynb` | Data preprocessing and dataset preparation |
| `notebooks/02_graph_construction.ipynb` | Distance-based and correlation-based graph construction |
| `notebooks/03_train_stgcn.ipynb` | STGCN training, graph ablation, multi-seed, missing-data, hidden-input training and hyperparameter-search results |
| `notebooks/03_train_dcrnn.ipynb` | DCRNN training and evaluation with the distance-based graph |
| `notebooks/04_train_dcrnn_correlation.ipynb` | DCRNN training and evaluation with the correlation-based graph |
| `notebooks/04_dcrnn_multiseed_distance.ipynb` | Multi-seed DCRNN runs on the distance-based graph (seeds 42, 123, 2024) |
| `notebooks/04_dcrnn_multiseed_correlation.ipynb` | Multi-seed DCRNN runs on the correlation-based graph (seeds 42, 123, 2024) |
| `notebooks/04_initial_data_exploration.ipynb` | Initial METR-LA data exploration and visualizations |
| `notebooks/05_train_lstm.ipynb` | LSTM baseline training and evaluation |
| `notebooks/06_historical_average_baseline.ipynb` | Historical-average baseline implementation and evaluation |
| `notebooks/07_compare_models.ipynb` | LSTM and historical-average baseline comparison |
| `results/` | Metrics, training histories, test predictions and plots for every model, missing-data results and the cross-model comparison |
| `results/dcrnn_graph_comparison_summary.csv` | Averaged DCRNN distance vs correlation metrics across seeds |

Setup (Python 3.10+):

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data: When running locally, place `metr-la.h5` and `adj_mx.pkl` in `data/raw/`. Both files are available in the shared Google Drive folder and are not tracked in GitHub because of their size.

The `adj_mx.pkl` sensor-graph file is also available from the DCRNN repository:
https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph

Run locally:

```
python -m src.data.pipeline        # writes data/processed/
jupyter notebook notebooks/        # open the required notebook
```

To run on Google Colab: add a shortcut to the shared Drive folder in My Drive named `CMPE258_TrafficGNN`, containing `data/raw/` with `metr-la.h5` and `adj_mx.pkl`. Open a notebook from GitHub in Colab and run all cells. Use a GPU runtime for model notebooks.

STGCN experiments (each command skips runs that already have results):

```
python -m src.models.stgcn.experiments run           # all STGCN variants and seeds (--limit N trains at most N runs)
python -m src.models.stgcn.experiments summary       # mean ± std table -> results/stgcn_summary.json
python -m src.models.stgcn.experiments tune          # hyperparameter search -> results/stgcn_tuning.json
python -m src.eval.missing_eval                      # missing-data tests for trained STGCNs -> results/missing_eval.json
python -m src.eval.compare                           # all models side by side -> results/comparison.json
```

## Saving results for the final comparison

Every model is compared on the same test windows by `src/eval/compare.py`. For a run named `<tag>` (for example `ha`, `lstm_s42`, `dcrnn_correlation_seed42`), save the test predictions in mph, shaped `[6832, 12, 207]` (test windows in time order, all 12 steps, sensors in METR-LA column order):

```python
np.savez_compressed(f"results/{tag}/test_predictions.npz", pred=pred_mph.astype(np.float32))
```

A run without saved predictions but with a `metrics.json` in the DCRNN format (`{"15_min": {"MAE": ..., "RMSE": ..., "MAPE": ...}, ...}`) still appears in the accuracy table.

Models that read recent speeds (LSTM, DCRNN) should also run the shared missing-data test, which hides exactly the same readings for every model:

```python
from src.eval.masked_inputs import mask_settings, masked_test_inputs, score, save_rows
rows = []
for mode, rate, seed in mask_settings():
    x = masked_test_inputs(mode, rate, seed)          # [6832, 12, 207], normalized, forward-filled, like test_X
    rows.append(score(predict_mph(x), mode, rate, seed))
save_rows(tag, rows)                                  # results/<tag>/missing_eval.json
```

The historical average does not use recent readings, so `compare.py` scores it on every missing-data setting automatically.

## Current Project Status

The repository now includes:

- METR-LA preprocessing and chronological train/validation/test splitting
- Distance-based and correlation-based sensor-graph construction
- STGCN training with distance, correlation, and no-graph variants
- STGCN multi-seed experiments using seeds 42, 43, and 44
- DCRNN training and evaluation using distance-based and correlation-based graphs
- DCRNN multi-seed experiments using seeds 42, 123, and 2024 for both graph types
- Missing-data robustness experiments with different masking and imputation strategies
- STGCN trained with hidden inputs (the final STGCN), for all three graph variants with three seeds
- A hyperparameter search for STGCN, judged on validation error
- A shared missing-data test and a cross-model comparison script that every model's results go through
- Evaluation using MAE, RMSE, and MAPE at 15-, 30-, and 60-minute horizons
- Saved metrics and training-history files under `results/`
- LSTM baseline training and evaluation with saved metrics and prediction visualizations
- Historical-average baseline evaluation with saved metrics and prediction visualizations

The LSTM baseline has been trained and evaluated at 15-, 30-, and 60-minute horizons. The historical-average baseline has been implemented and evaluated at 15-, 30-, and 60-minute horizons. DCRNN multi-seed distance and correlation results are available and summarized below.

## Updates Since the Original Proposal

The original proposal identified DCRNN as the primary graph-based model and listed STGCN as an optional extension. During development, the team implemented substantial STGCN experiments, including distance-based, correlation-based and no-graph variants, multi-seed evaluation and missing-data experiments. DCRNN was implemented and evaluated on both distance-based and correlation-based graphs, including multi-seed runs with seeds 42, 123, and 2024. The LSTM baseline was implemented and evaluated. The historical-average baseline was implemented and evaluated using the same data split, normalization procedure, and evaluation metrics.

## Preliminary Experimental Results

The STGCN models were evaluated at 15-, 30- and 60-minute forecasting horizons using MAE, RMSE and MAPE. Results were averaged across random seeds 42, 43 and 44.

| Graph variant | 15-minute MAE | 30-minute MAE | 60-minute MAE |
|---|---:|---:|---:|
| Distance-based graph | 2.799 | 3.223 | 3.737 |
| Correlation-based graph | 2.808 | 3.220 | 3.738 |
| No graph connections | 2.948 | 3.450 | 3.998 |

The distance-based and correlation-based graph models performed better than the no-graph model. The performance difference was more noticeable at the 60-minute horizon, showing that spatial relationships between traffic sensors are useful for longer-term forecasting.

Additional experiments evaluated model performance under missing sensor readings and compared different imputation strategies. Detailed metrics and training histories are stored in the `results/` directory.

### Final STGCN (Check-in 4)

The final STGCN fills missing inputs with the training mean plus a missing-indicator channel, and is trained with extra inputs hidden in every batch (`missing.train_mask`). Test MAE in mph, mean ± standard deviation over seeds 42, 43 and 44:

| Graph variant | 15-minute MAE | 30-minute MAE | 60-minute MAE | Dead sensors, 30% switched off |
|---|---:|---:|---:|---:|
| Correlation-based graph (final) | 2.718 ± 0.010 | 3.081 ± 0.014 | 3.531 ± 0.020 | 3.45 |
| Distance-based graph | 2.745 ± 0.025 | 3.124 ± 0.026 | 3.576 ± 0.024 | 3.49 |
| No graph connections | 2.914 ± 0.059 | 3.340 ± 0.060 | 3.805 ± 0.069 | 4.20 |

Training with hidden inputs lowers the normal test error by 3–4% and cuts the error on switched-off sensors from 5.1–6.4 mph to about 3.3–3.5 mph. A hyperparameter search over seven single changes (`results/stgcn_tuning.json`) found nothing better than the original settings by more than 0.01 mph on validation.


### DCRNN Results

The DCRNN models were evaluated at 15-, 30- and 60-minute forecasting horizons using MAE, RMSE and MAPE. Results were averaged across random seeds 42, 123 and 2024 for both the distance-based and correlation-based sensor graphs.

| Graph variant | 15-minute MAE | 30-minute MAE | 60-minute MAE |
|---|---:|---:|---:|
| Correlation-based graph | 2.937 | 3.486 | 4.321 |
| Distance-based graph | 2.963 | 3.550 | 4.445 |

| Graph variant | Horizon | MAE | RMSE | MAPE |
|---|---|---:|---:|---:|
| Correlation-based graph | 15 minutes | 2.937 ± 0.007 | 5.557 | 7.88% |
| Correlation-based graph | 30 minutes | 3.486 ± 0.010 | 6.771 | 9.91% |
| Correlation-based graph | 60 minutes | 4.321 ± 0.036 | 8.402 | 12.80% |
| Distance-based graph | 15 minutes | 2.963 ± 0.006 | 5.663 | 7.92% |
| Distance-based graph | 30 minutes | 3.550 ± 0.020 | 7.003 | 10.18% |
| Distance-based graph | 60 minutes | 4.445 ± 0.051 | 8.790 | 13.75% |

The correlation-based DCRNN performed slightly better than the distance-based DCRNN at all three horizons. Error increased with longer forecasting horizons for both graph variants. Per-seed metrics are stored under `results/dcrnn_distance_seed*/` and `results/dcrnn_correlation_seed*/`, and the averaged comparison is available at `results/dcrnn_graph_comparison_summary.csv`.

### Historical-Average Baseline Results

The historical-average baseline was evaluated using the same forecasting horizons and metrics as the other models.

| Horizon | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 15 minutes | 4.1858	| 7.8507 | 13.0299% |
| 30 minutes | 4.1869 | 7.8524 | 13.0348% |
| 60 minutes | 4.1893 | 7.8559 | 13.0447% |

The historical-average baseline produced nearly constant performance across the three forecasting horizons. It provides a simple non-neural reference for comparison with the LSTM, STGCN, and DCRNN models.

The prediction visualization is available at `results/historical_average_prediction_comparison.png`, and the detailed metrics are stored in `results/historical_average_metrics.json`.


### LSTM Baseline Results

The LSTM baseline was evaluated using the same forecasting horizons and metrics.

| Horizon | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 15 minutes | 3.8104 | 6.5285 | 10.6345% |
| 30 minutes | 3.9324 | 6.8425 | 11.1449% |
| 60 minutes | 4.2159 | 7.4250 | 12.2251% |

The LSTM error increased as the forecasting horizon became longer. The LSTM performed better than the historical-average baseline at all three horizons.

The LSTM loss curve and prediction visualization are available at `results/lstm_loss_curve.png` and `results/lstm_prediction_comparison.png`. The detailed metrics are stored at `results/lstm_metrics.json`.

The baseline comparison results for the LSTM and historical-average models are available at `results/baseline_comparison.json`.

## Team Contributions 

- **Lavya Sri Bonagiri:** Implemented and evaluated the LSTM and historical-average baselines using the same METR-LA data split, normalization procedure, and evaluation metrics. The LSTM used the previous 12 traffic-speed measurements to predict the next 12 steps. The historical-average baseline used training-period averages for the corresponding day of week and time of day. Both baselines were evaluated at 15-, 30-, and 60-minute horizons using MAE, RMSE, and MAPE. Generated the LSTM loss curve, actual-versus-predicted plots, and historical-average prediction visualization, and added the notebooks and result artifacts to the project repository.
  
- **Mansi Gupta:** Implemented and organized the data-preprocessing pipeline, including chronological train/validation/test splitting, normalization, sliding-window preparation, missing-value handling and dataset preparation for model training. She also implemented and trained the STGCN model, conducted distance-based, correlation-based and no-graph comparisons, ran multi-seed experiments using seeds 42, 43, and 44, and contributed to missing-data and imputation experiments.

- **Megha Gangal:** Worked on traffic-sensor graph construction, including distance-based and correlation-based graphs. Implemented and trained the DCRNN model on both the distance-based and correlation-based sensor graphs, ran multi-seed experiments using seeds 42, 123, and 2024, and compared the two graph variants across 15-, 30-, and 60-minute horizons. Also contributed to model evaluation, training experiments, evaluation utilities, and preliminary-result analysis.

- **All team members:** Contributed to project discussions, report coordination, code review, final documentation and presentation preparation.

## Future Work

Future work will focus on completing the final cross-model comparison using compatible training and evaluation settings for the historical-average, LSTM, STGCN, and DCRNN models across all forecasting horizons. Further work may also include common missing-data evaluations for all models and longer DCRNN training with early stopping where additional compute is available.
