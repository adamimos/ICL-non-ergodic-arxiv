# AGENT Instructions

This repository provides utilities for analyzing the arXiv dataset. When modifying the code or scripts, please follow these guidelines.

## Style
- Write Python code that conforms to standard PEP8 formatting.
- Prefer descriptive variable names and include docstrings for public functions.

## Testing
- Run the unit test suite using Python's built-in `unittest` module:
  ```bash
  python -m unittest
  ```
- Verify basic script functionality by running the analysis script help command:
  ```bash
  python scripts/analyze_dataset.py --help
  ```
  The command should display usage information without errors.

## Dependency Management
- Install dependencies using `uv pip install -r requirements.txt`.

