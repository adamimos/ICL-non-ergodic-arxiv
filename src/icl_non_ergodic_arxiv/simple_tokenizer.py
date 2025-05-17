"""Utility functions for lightweight tokenization."""

from typing import List


def count_whitespace_tokens(text: str) -> int:
    """Count tokens by splitting on whitespace.

    Parameters
    ----------
    text: str
        The input string to tokenize.

    Returns
    -------
    int
        Number of whitespace-separated tokens.
    """
    # `split()` will ignore multiple consecutive spaces and leading/trailing spaces.
    return len(text.split())
