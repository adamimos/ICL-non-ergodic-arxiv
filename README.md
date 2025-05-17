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

