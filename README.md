# LightKG (fork) — Amazon Review Data (2018)

This repository is a **fork** of the official implementation of **LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture**. It adds support and pre-processed data for the [Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) datasets, on top of RecBole.

## Table of Contents
- [Requirements & Environment Setup](#requirements--environment-setup)
- [In a Nutshell](#in-a-nutshell)
- [Results](#results)
- [Credits](#credits)

## Requirements & Environment Setup

PyTorch runs on GPU (CUDA), along with the `torch-scatter` and `torch-sparse` extensions. Set up a fresh Python environment before installing.

**Core environment:**
1. [Python](https://www.python.org/downloads/release/python-3919/) == 3.9.19
2. [NumPy](https://numpy.org/) < 1.24.0 
3. [PyTorch](https://pytorch.org/) == 2.0.0
4. [RecBole](https://recbole.io/) == 1.1.1
5. [torch-scatter](https://docs.pytorch.org/docs/2.0/generated/torch.scatter.html)
6. [torch-sparse](https://docs.pytorch.org/docs/2.0/sparse.html)
7. [lightgbm](https://lightgbm.org/)
8. [ray\[tune\]](https://docs.ray.io/en/latest/tune/index.html)
9. [pydantic](https://pydantic.dev/docs/validation/latest/get-started/)

## In a Nutshell

To start training the model on the Luxury Beauty dataset, simply run:

```bash
python main.py --dataset Amazon-luxury-beauty
```

You can also run any of the paper's original datasets:

```bash
python main.py --dataset lastfm
python main.py --dataset Amazon-book
python main.py --dataset ml-1m
python main.py --dataset book-crossing
```

## Results

Performance evaluated on Amazon Luxury Beauty (5-core filter, 80/10/10 split, fixed seed 2020):


| Metric @10 | Value  |
|------------|--------|
| Recall     | 0.2633 |
| MRR        | 0.2354 |
| NDCG       | 0.2394 |
| Hit Rate   | 0.2774 |
| Precision  | 0.0365 |


## Credits
- Official implementation of [LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture](https://github.com/1371149/LightKG/tree/main).
- Framework built on top of [RecBole](https://recbole.io/).
- Dataset sourced from [Amazon Review Data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).