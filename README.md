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
| `src/data/` | Data loading, chronological splitting, normalization, missing-value handling, and sliding-window generation |
| `src/eval/` | MAE, RMSE, MAPE, and missing-data evaluation utilities |
| `src/models/stgcn/` | STGCN model, graph construction, training, and experiment utilities |
| `notebooks/01_data_preprocess.ipynb` | Data preprocessing and dataset preparation |
| `notebooks/02_graph_construction.ipynb` | Distance-based and correlation-based graph construction |
| `notebooks/03_train_stgcn.ipynb` | STGCN training, graph ablation, multi-seed, and missing-data experiments |
| `notebooks/03_train_dcrnn.ipynb` | DCRNN implementation and evaluation |
| `notebooks/04_initial_data_exploration.ipynb` | Initial METR-LA data exploration and visualizations |
| `notebooks/05_train_lstm.ipynb` | LSTM baseline training and evaluation |
| `notebooks/06_historical_average_baseline.ipynb` | Historical-average baseline implementation and evaluation |
| `results/` | STGCN, DCRNN, LSTM, and historical-average metrics, plots, and training histories |

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

## Current Project Status

The repository now includes:

- METR-LA preprocessing and chronological train/validation/test splitting
- Distance-based and correlation-based sensor-graph construction
- STGCN training with distance, correlation, and no-graph variants
- Multi-seed experiments using seeds 42, 43, and 44
- DCRNN training notebooks and evaluation using distance-based and correlation-based graphs
- Missing-data robustness experiments with different masking and imputation strategies
- Evaluation using MAE, RMSE, and MAPE at 15-, 30-, and 60-minute horizons
- Saved metrics and training-history files under `results/`
- LSTM baseline training and evaluation with saved metrics and prediction visualizations
- Historical-average baseline evaluation with saved metrics and prediction visualizations

The LSTM baseline has been trained and evaluated at 15-, 30-, and 60-minute horizons. The historical-average baseline has been implemented and evaluated at 15-, 30-, and 60-minute horizons.

## Updates Since the Original Proposal

The original proposal identified DCRNN as the primary graph-based model and listed STGCN as an optional extension. During development, the team implemented substantial STGCN experiments, including distance-based, correlation-based and no-graph variants, multi-seed evaluation and missing-data experiments. DCRNN was also implemented and evaluated. The LSTM baseline was implemented and evaluated. The historical-average baseline was implemented and evaluated using the shared preprocessing and evaluation pipeline.

## Preliminary Experimental Results

The STGCN models were evaluated at 15-, 30- and 60-minute forecasting horizons using MAE, RMSE and MAPE. Results were averaged across random seeds 42, 43 and 44.

| Graph variant | 15-minute MAE | 30-minute MAE | 60-minute MAE |
|---|---:|---:|---:|
| Distance-based graph | 2.799 | 3.223 | 3.737 |
| Correlation-based graph | 2.808 | 3.220 | 3.738 |
| No graph connections | 2.948 | 3.450 | 3.998 |

The distance-based and correlation-based graph models performed better than the no-graph model. The performance difference was more noticeable at the 60-minute horizon, showing that spatial relationships between traffic sensors are useful for longer-term forecasting.

Additional experiments evaluated model performance under missing sensor readings and compared different imputation strategies. Detailed metrics and training histories are stored in the `results/` directory.

### Historical-Average Baseline Results

The historical-average baseline was evaluated using the same forecasting horizons and metrics as the other models.

| Horizon | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 15 minutes | 4.2846 | 7.9921 | 13.2415% |
| 30 minutes | 4.5184 | 8.3833 | 13.8754% |
| 60 minutes | 5.1756 | 9.5160 | 15.7827% |

The historical-average baseline error increased for longer forecasting horizons. It provides a simple non-neural reference for comparison with the LSTM, STGCN, and DCRNN models.

The prediction visualization is available at `results/historical_average_prediction_comparison.png`, and the detailed metrics are stored in `results/historical_average_metrics.json`.

## Team Contributions 

- **Lavya Sri Bonagiri:** Implemented and evaluated the LSTM and historical-average baselines using the shared METR-LA preprocessing and evaluation pipeline. The LSTM used the previous 12 traffic-speed measurements to predict the next 12 steps. The historical-average baseline used training-period averages for the corresponding day of week and time of day. Both baselines were evaluated at 15-, 30-, and 60-minute horizons using MAE, RMSE, and MAPE. Generated the LSTM loss curve, actual-versus-predicted plots, and historical-average prediction visualization, and added the notebooks and result artifacts to the project repository.
  
- **Mansi Gupta:** Implemented and organized the data-preprocessing pipeline, including chronological train/validation/test splitting, normalization, sliding-window preparation, missing-value handling and dataset preparation for model training. She also implemented and trained the STGCN model, conducted distance-based, correlation-based and no-graph comparisons, ran multi-seed experiments using seeds 42, 43, and 44, and contributed to missing-data and imputation experiments.

- **Megha Gangal:** Worked on traffic-sensor graph construction, including distance-based and correlation-based graphs. She implemented and trained the DCRNN model using the distance-based sensor graph and contributed to model evaluation, training experiments and comparison of graph-based forecasting approaches. She also contributed to evaluation utilities and preliminary-result analysis.

- **All team members:** Contributed to project discussions, report coordination, code review, final documentation and presentation preparation.

## Future Work

Future work will focus on completing the final cross-model comparison using compatible training and evaluation settings. The comparison will include the historical-average, LSTM, STGCN, and DCRNN models across all forecasting horizons. Further work may also include longer DCRNN training with early stopping and common missing-data evaluations for all models.
