"""Model training script.

This script loads a prepared dataset from the ``data/`` directory and
fine-tunes a Hugging Face transformers model on it.
"""

import argparse
import os


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a transformers model")
    parser.add_argument(
        "--model_name",
        default="distilbert-base-uncased",
        help="Model checkpoint name or path",
    )
    parser.add_argument(
        "--data_dir", default="data", help="Directory containing prepared data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Optimizer learning rate"
    )
    return parser.parse_args()


def main() -> None:
    """Run a simple training loop using the ``Trainer`` API.

    The function loads a dataset saved in ``data/`` via ``load_from_disk``,
    tokenizes the ``text`` field, initializes a sequence classification model,
    and fine-tunes it for the requested number of epochs.
    """

    args = parse_args()

    from datasets import load_from_disk, DatasetDict
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    dataset = load_from_disk(args.data_dir)
    train_ds = dataset["train"] if isinstance(dataset, DatasetDict) else dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized_ds = train_ds.map(tokenize, batched=True)

    num_labels = len(train_ds.features["label"].names)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.data_dir, "model"),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
