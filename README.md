# ICL-non-ergodic-arxiv

## Dataset analysis

The `scripts/analyze_dataset.py` utility loads the `ccdv/arxiv-classification` dataset and reports token usage per arXiv category.
Each dataset entry provides the full paper in the `text` field and an integer `label` corresponding to one of eleven arXiv subsections.

### Requirements

Install the dependencies with:

```bash
pip install datasets transformers
```

### Usage

Run the analysis script specifying how many of the largest categories to display:

```bash
python scripts/analyze_dataset.py --top_n 5
```

The script downloads the dataset, tokenizes the `text` field with the chosen tokenizer (default: `bert-base-uncased`), and prints the top categories ordered by total token count. Use `--split` or `--tokenizer` to customize the dataset split or tokenizer.

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