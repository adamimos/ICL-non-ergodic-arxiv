# ICL Non-Ergodic Arxiv

This repository explores non-ergodic behavior in in-context learning. The main experiment studies how the number of ergodic components scales across different categories of arXiv content.

## Ergodic-Component Scaling

The goal is to measure how in-context learning performance varies when sampling from different arXiv categories. We focus on a subset of categories from the `cs` and `stat` domains, e.g. **cs.CL**, **cs.LG**, and **stat.ML**.

## Dataset

We use the `ccdv/arxiv-classification` dataset with the `no_ref` configuration. Full texts are tokenized and truncated to fit model context windows.

## Project Layout

```
src/      # model and training code
scripts/  # helper scripts for data processing and running experiments
data/     # preprocessed datasets and experiment outputs
```

To run a basic experiment, execute one of the scripts in `scripts/` from the repository root:

```bash
python scripts/train.py
```

Install dependencies with:

```bash
uv pip install -r requirements.txt
```
