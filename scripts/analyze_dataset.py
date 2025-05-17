import argparse
from collections import defaultdict



def parse_args():
    parser = argparse.ArgumentParser(description="Analyze token counts per arXiv category")
    parser.add_argument(
        "--top_n",
        type=int,
        default=11,
        help="Number of categories to display ordered by total token volume",
    )
    parser.add_argument(
        "--tokenizer",
        default="bert-base-uncased",
        help="Tokenizer name or path for tokenizing the text field",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to analyze (default: train)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import heavy dependencies lazily so the help command works even if they
    # are not installed.
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load the dataset. This will download the split if not already present.
    data = load_dataset("ccdv/arxiv-classification", "no_ref", split=args.split)

    # Map integer labels to their string names for readability
    label_names = data.features["label"].names

    # Initialize tokenizer (replace with your preferred tokenizer if desired).
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Aggregate token counts per category using the full text field
    token_counts = defaultdict(int)
    for row in data:
        text = row.get("text", "")
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        label_id = row.get("label")
        if label_id is not None and 0 <= label_id < len(label_names):
            label = label_names[label_id]
            token_counts[label] += n_tokens

    # Sort by token count in descending order.
    sorted_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    if args.top_n is not None:
        sorted_counts = sorted_counts[: args.top_n]

    print(f"Top {len(sorted_counts)} categories by token volume in split '{args.split}':")
    for cat, count in sorted_counts:
        print(f"{cat}\t{count}")


if __name__ == "__main__":
    main()

