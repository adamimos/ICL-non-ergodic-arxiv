"""Utilities for preparing the arXiv dataset.

The script downloads the raw data, tokenizes the ``text`` field and
saves the processed split under the ``data/`` directory.  The default
configuration uses the ``ccdv/arxiv-classification`` dataset with the
``no_ref`` configuration and a BERT tokenizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def download_dataset(split: str) -> Dataset:
    """Download a split of the arXiv dataset.

    Parameters
    ----------
    split:
        Name of the dataset split to download, e.g. ``"train"``.

    Returns
    -------
    Dataset
        The loaded Hugging Face dataset split.
    """

    return load_dataset("ccdv/arxiv-classification", "no_ref", split=split)


def preprocess_dataset(
    dataset: Dataset,
    tokenizer_name: str,
    max_length: Optional[int] = 512,
) -> Dataset:
    """Tokenize and filter a dataset.

    Parameters
    ----------
    dataset:
        The raw dataset to preprocess.
    tokenizer_name:
        Hugging Face tokenizer identifier.
    max_length:
        If provided, sequences are truncated to at most this many tokens.

    Returns
    -------
    Dataset
        The processed dataset containing an ``input_ids`` column.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(example: dict) -> dict:
        tokens = tokenizer.encode(
            example["text"],
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
        )
        return {"input_ids": tokens}

    dataset = dataset.filter(lambda x: x.get("text") and x["text"].strip())
    return dataset.map(tokenize)


def main() -> None:
    """Download, preprocess and store the arXiv dataset."""

    split = "train"
    tokenizer_name = "bert-base-uncased"
    output_dir = Path("data") / f"prepared_{split}"

    raw_data = download_dataset(split)
    processed_data = preprocess_dataset(raw_data, tokenizer_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_data.save_to_disk(str(output_dir))


if __name__ == "__main__":
    main()
