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
| `configs/metr_la.yaml` | Shared settings: input window, horizons, split ratios, normalization, graph choice, STGCN hyperparameters |
| `src/data/` | Loading, chronological split, per-sensor normalization, missing-value handling, sliding-window datasets |
| `src/models/stgcn/` | STGCN: graph construction and model |
| `notebooks/01_data_preprocess.ipynb` | Data exploration and preprocessing (run this first) |
| `notebooks/03_train_stgcn.ipynb` | STGCN forward pass and training smoke test |

Setup (Python 3.10+):

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Data: `metr-la.h5` and `adj_mx.pkl` in `data/raw/`. Both files are in the Drive folder mentioned above

 `adj_mx.pkl` is also at https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph.

I also added dataset in data/raw folder to run locally.

Run locally:

```
python -m src.data.pipeline        # writes data/processed/
jupyter notebook notebooks/        # open 01, then 03
```

To run on Google Colab: add a shortcut to the shared Drive folder in My Drive named `CMPE258_TrafficGNN` (with `raw/` inside it), open a notebook from GitHub in Colab, and run all cells. The first cell mounts Drive, clones this repository and installs the extra packages. Use a GPU runtime for model notebooks.

data pipeline and the STGCN model are implemented. The STGCN notebook runs a forward pass and a short training smoke test. 

Will work on full training, evaluation and the other models come in later check-ins.
