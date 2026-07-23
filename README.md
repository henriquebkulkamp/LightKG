## LightKG (fork) — with support for Amazon Review Data (2018) datasets

This repository is a **fork** of LightKG — the official implementation of the paper *"LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture"*. This fork adds a custom dataset preparation pipeline (`preparar_dataset.py`) that converts raw dumps from the **Amazon Review Data (2018)** into the Knowledge Graph format expected by RecBole, making it possible to train LightKG on Amazon product categories beyond the datasets originally used in the paper.

Tested on the **Luxury Beauty** category, with inference results reported below.

---

## Table of Contents

- [About LightKG](#about-lightkg)
- [What this fork adds](#what-this-fork-adds)
- [Environment requirements](#environment-requirements)
- [Preparing an Amazon dataset](#preparing-an-amazon-dataset)
- [Running training](#running-training)
- [Results](#results)
- [Project structure](#project-structure)
- [Credits](#credits)

---

## About LightKG

LightKG is a GNN-based (Graph Neural Network) recommender model that leverages a Knowledge Graph to improve recommendation quality, especially in sparse-interaction scenarios. The architecture is a simplified version of prior KG-aware models, prioritizing training efficiency without sacrificing accuracy.

Key characteristics of the implementation (`LightKG.py`):

- Builds a **CKG (Collaborative Knowledge Graph)**, merging the user-item interaction graph with the knowledge graph (entities such as brand, category, etc.).
- Propagates embeddings through multiple sparse graph-convolution layers (`torch_sparse.spmm`), aggregating across layers by mean pooling (LightGCN-style).
- Degree-based normalization of nodes (`1/sqrt(degree)`).
- Loss function combining **BPR loss** (Bayesian Personalized Ranking) with an optional **contrastive loss** (`cos_loss`), computed separately for users and items from a similarity matrix derived from the CKG.
- Built on top of the [RecBole](https://recbole.io/) framework, extending `KnowledgeRecommender`.

---

## What this fork adds

The original repository already supports the ML-1M, Last.FM, Book-Crossing, and Amazon-book datasets. This fork adds:

**`preparar_dataset.py`** — a script that generates, from raw [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html) files, the four files RecBole needs to train a KG-aware model:

| File | Content |
|---|---|
| `.train.inter` / `.valid.inter` / `.test.inter` | User-item interactions (`user_id`, `item_id`, `rating`, `timestamp`), split 80/10/10 |
| `.kg` | Knowledge graph triples (`head`, `relation`, `tail`) |
| `.link` | Mapping between `item_id` and `entity_id` in the graph |

### Script pipeline

1. **Reading and 5-core filtering of interactions**
   Reads the reviews file (`Luxury_Beauty.json`), removes duplicate user-item pairs, and applies **iterative k-core filtering** (k=5 by default): users and items with fewer than 5 interactions are removed, repeating the process until convergence. This ensures a minimum density for the model to learn meaningful representations.

2. **Train/validation/test split**
   Shuffles the interactions (fixed seed = 2020, for reproducibility) and splits them into 80% train / 10% validation / 10% test.

3. **Knowledge Graph extraction**
   From the metadata file (`meta_Luxury_Beauty.json`), extracts triples only for items that survived the 5-core filter, with the following relations:
   - `brand_of` — the product's brand
   - `category_of` — product categories (can generate multiple triples per item)
   - `has_form` — item format/formulation (extracted from `details.Item Form` or `details.Formulation`)
   - `also_bought` — item-item "also bought" relation (restricted to items within the 5-core set)
   - `also_viewed` — item-item "also viewed" relation (restricted to items within the 5-core set)

   Entity text is sanitized (`sanitize_entity`) to avoid ID collisions in RecBole (lowercased, special characters replaced with `_`).

   If the metadata file yields no valid triples, the script applies a **safety fallback**, creating a generic `is_a → luxury_beauty_item` relation for all items, so the pipeline doesn't break due to a missing KG.

4. **Generating the `.link` file**
   Since item IDs already match entity IDs (`asin`) in this case, the mapping is direct (`item_id == entity_id`).

### How to run the preparation step

```bash
# Download the raw Amazon Review Data (2018) files and place them in the project root:
#   Luxury_Beauty.json
#   meta_Luxury_Beauty.json

python preparar_dataset.py
```

This generates the folder `dataset/Amazon-luxury-beauty/` with the four files (`.train.inter`, `.valid.inter`, `.test.inter`, `.kg`, `.link`), ready for RecBole.

> To use another Amazon Review Data category, simply change the `DATASET_NAME`, `REVIEWS_FILE`, and `META_FILE` constants at the top of the script.

You will also need to create the corresponding YAML config file (`./yaml/Amazon-luxury-beauty_LightKG.yaml`), following the pattern of the other YAML files in the original project.

---

## Environment requirements

Tested on Python 3.9 / Ubuntu 20.04.

- [PyTorch](https://pytorch.org/) == 2.0.0
- [RecBole](https://recbole.io/) == 1.1.1
- [LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)
- [XGBoost](https://github.com/dmlc/xgboost)
- [Ray](https://www.ray.io/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter/tree/master)
- `torch_sparse`
- `pandas`, `numpy`

---

## Preparing an Amazon dataset

Quick summary (see previous section for details):

```bash
python preparar_dataset.py
```

Expected console output:

```
==================================================
 Creating RecBole KG Dataset (5-Core) for: Amazon-luxury-beauty
==================================================

[1/4] Reading and applying 5-core filtering to interactions...
   [Iteration 1] Interactions: ... | Users: ... | Items: ...
   ...

[2/4] Extracting Knowledge Graph entities and edges...
 -> Saved Amazon-luxury-beauty.kg with N heterogeneous triples.

[3/4] Generating alignment file (.link)...

==================================================
 DONE! Generated files:
==================================================
```

---

## Running training

```bash
python main.py --dataset Amazon-luxury-beauty
```

Datasets originally supported by the repository:

```bash
python main.py --dataset lastfm
python main.py --dataset Amazon-book
python main.py --dataset ml-1m
python main.py --dataset book-crossing
```

What `main.py` does:
1. Loads the config from the YAML file matching the dataset (`./yaml/<dataset>_LightKG.yaml`).
2. Creates the dataset and dataloaders via RecBole's `create_dataset` / `data_preparation`.
3. Instantiates the `LightKG` model.
4. Trains with `KGTrainer`, validating along the way and evaluating on the test set at the end.
5. Logs the best validation results and the final test results.

---

## Results

LightKG trained on the **Amazon Luxury Beauty** dataset (Amazon Review Data, 2018), prepared with the 5-core pipeline described above:

```
================ INFERENCE RESULTS ================
OrderedDict([('recall@10', 0.2633), ('mrr@10', 0.2354), ('ndcg@10', 0.2394),
             ('hit@10', 0.2774), ('precision@10', 0.0365)])
```

| Metric @10 | Value |
|---|---|
| Recall | 0.2633 |
| MRR | 0.2354 |
| NDCG | 0.2394 |
| Hit Rate | 0.2774 |
| Precision | 0.0365 |

> Results obtained with 5-core filtering and an 80/10/10 split, using a fixed seed (2020) for reproducible data splitting.

---

## Project structure

```
.
├── LightKG.py                 # Model implementation
├── main.py                    # Training/evaluation CLI script
├── preparar_dataset.py        # Amazon → RecBole KG format preparation pipeline
├── model/                     # Baseline models (CFKG, KGAT, CKE, RippleNet, KGIN, KGCN, LightGCN, MCCLK...)
├── yaml/                      # Per-dataset config files
└── dataset/
    └── Amazon-luxury-beauty/  # Generated by preparar_dataset.py
        ├── Amazon-luxury-beauty.train.inter
        ├── Amazon-luxury-beauty.valid.inter
        ├── Amazon-luxury-beauty.test.inter
        ├── Amazon-luxury-beauty.kg
        └── Amazon-luxury-beauty.link
```

---

## Credits

- Original repository: official implementation of the paper **LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture**.
- Recommendation framework: [RecBole](https://recbole.io/).
- Dataset: [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html).
- Some baseline models in the original project come from RecBole; KGRec and DiffKG were obtained from [SSLRec](https://github.com/HKUDS/SSLRec).
- CL-SDKG is not open-source; obtaining it requires directly contacting the original authors.

### Fork notes

- This fork only adds the data preparation pipeline for Amazon Review Data (2018) categories and documentation of the results obtained with it. The model architecture (`LightKG.py`) remains the original implementation.