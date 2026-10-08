Traffic-Speed Forecasting Using Graph Neural Networks
Deep Learning Course 258-03, Fall 2026, San José State University


----------

Team Members: Lavya Sri Bonagiri(017552834), Mansi Gupta (020088497), Megha Gangal (020096869)


----------

Dataset :

https://drive.google.com/drive/folders/1zijctqUvd5rywZyIUK8cRHI3JtknPYhq?usp=sharing
----------


Accurate traffic prediction can help drivers choose better routes, reduce traffic congestion and improve traffic-light control. However, many traditional time-series models only study how traffic changes over time and do not consider how nearby roads affect one another.
This project will predict traffic speed at each sensor 15, 30 and 60 minutes into the future, corresponding to 3, 6 and 12 steps ahead, respectively, because the dataset uses a five-minute sampling interval. We will represent the traffic sensor network as a graph, where each sensor is a node and edges encode road connectivity and the distance between sensors. This structure allows the model to learn both how traffic changes over time and how conditions at one sensor may affect connected sensors.

