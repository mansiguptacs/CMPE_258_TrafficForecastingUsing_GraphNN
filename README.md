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
This project will predict traffic speed at each sensor 15, 30 and 60 minutes into the future, corresponding to 3, 6 and 12 steps ahead, respectively, because the dataset uses a five-minute sampling interval. We will represent the traffic sensor network as a graph, where each sensor is a node and edges encode road connectivity and the distance between sensors. This structure allows the model to learn both how traffic changes over time and how conditions at one sensor may affect connected sensors.


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
| `notebooks/03_train_dcrnn.ipynb` | DCRNN implementation and evaluation on the distance-based graph |
| `notebooks/04_train_dcrnn_correlation.ipynb`, `notebooks/04_dcrnn_multiseed_correlation.ipynb` | DCRNN on the correlation-based graph, single and multi-seed |
| `notebooks/04_dcrnn_multiseed_distance.ipynb` | DCRNN on the distance-based graph, multi-seed |
| `notebooks/04_initial_data_exploration.ipynb` | Initial METR-LA data exploration and visualizations |
| `results/` | Model metrics, training histories, multi-seed summaries, missing-data results and the cross-model comparison |

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
- Multi-seed experiments using seeds 42, 43, and 44
- DCRNN training and evaluation on the distance-based and correlation-based graphs, three seeds each
- Missing-data robustness experiments with different masking and imputation strategies
- STGCN trained with hidden inputs (the final STGCN), for all three graph variants with three seeds
- A hyperparameter search for STGCN, judged on validation error
- A shared missing-data test and a cross-model comparison script that every model's results go through
- Evaluation using MAE, RMSE, and MAPE at 15-, 30-, and 60-minute horizons
- Saved metrics and training-history files under `results/`

Historical-average and LSTM baselines are still planned or under development.

## Updates Since the Original Proposal

The original proposal identified DCRNN as the primary graph-based model and listed STGCN as an optional extension. During development, the team implemented substantial STGCN experiments, including distance-based, correlation-based and no-graph variants, multi-seed evaluation and missing-data experiments. DCRNN was also implemented and evaluated. The historical-average and LSTM baselines remain under development.

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

DCRNN, trained for 10 epochs per run with seeds 42, 123 and 2024, reaches 2.94, 3.49 and 4.32 mph MAE at 15, 30 and 60 minutes on the correlation-based graph, slightly better than on the distance-based graph (2.96, 3.55 and 4.45). Both graph models therefore do slightly better with the correlation-based graph. DCRNN predictions are not saved yet, so `compare.py` reads its accuracy from each run's `metrics.json`, and DCRNN is not yet in the traffic-state and missing-data comparisons.

## Team Contributions Through Check-in 3

- **Lavya Sri Bonagiri:** For this check-in, she updated and organized the project documentation for Check-in 3. Her work included updating the README with the project status, repository instructions, preliminary experimental results, project updates and team contribution information. She also helped organize the written project documentation for submission.

- **Mansi Gupta:** Implemented and organized the data-preprocessing pipeline, including chronological train/validation/test splitting, normalization, sliding-window preparation, missing-value handling and dataset preparation for model training. She also implemented and trained the STGCN model, conducted distance-based, correlation-based and no-graph comparisons, ran multi-seed experiments using seeds 42, 43, and 44, and contributed to missing-data and imputation experiments.

- **Megha Gangal:** Worked on traffic-sensor graph construction, including distance-based and correlation-based graphs. She implemented and trained the DCRNN model on the distance-based and correlation-based graphs, three seeds each, and contributed to model evaluation, training experiments and comparison of graph-based forecasting approaches. She also contributed to evaluation utilities and preliminary-result analysis.

- **All team members:** Contributed to project discussions, report coordination, code review, final documentation and presentation preparation.

## Future Work

The team plans to implement and evaluate the historical-average and LSTM baseline models, save DCRNN test predictions in the shared format, and run every model through `src/eval/compare.py` and the shared missing-data test, so that all four models are compared with the same forecasting horizons and evaluation metrics.
