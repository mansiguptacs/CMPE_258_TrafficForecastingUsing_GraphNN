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
| `results/` | Model metrics, training histories, multi-seed results, and missing-data results |

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

To run on Google Colab: add a shortcut to the shared Drive folder in My Drive named `CMPE258_TrafficGNN` (with `raw/` inside it), open a notebook from GitHub in Colab, and run all cells. The first cell mounts Drive, clones this repository and installs the extra packages. Use a GPU runtime for model notebooks.

## Current Project Status

The repository now includes:

- METR-LA preprocessing and chronological train/validation/test splitting
- Distance-based and correlation-based sensor-graph construction
- STGCN training with distance, correlation, and no-graph variants
- Multi-seed experiments using seeds 42, 43, and 44
- DCRNN training notebook and evaluation using the distance-based graph
- Missing-data robustness experiments with different masking and imputation strategies
- Evaluation using MAE, RMSE, and MAPE at 15-, 30-, and 60-minute horizons
- Saved metrics and training-history files under `results/`

Historical-average and LSTM baselines are still planned or under development.
