import itertools
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from experiment_utils import (
    save_csv, set_seed, make_splits, score_predictions, summarize, compare_systems,
    save_settings, save_split_indices, save_training_history, save_diagnostics,
    save_predictions,
)



REPRESENTATIONS = ("indicator", "bow", "tfidf")
MODELS = ("loglinear", "mlp1", "mlp2")
TOPIC = "stance_atheism"
SEEDS = [0]
NUM_FOLDS = 5
NUM_EPOCHS = 25
HIDDEN_DIM = 128
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ROPE = 0.01
DEVICE = "cpu"
INNER_FRACTION = 0.2
OUTPUT_DIR = Path("results/atheism_exp01_epoch25")


def make_encoder(name, tokenizer, training_texts):
    """Fit corpus-dependent state on the inner fitting subset only."""
    from rep_spaces import TokenIndicator, BagOfWords, TF_IDF
    if name == "tfidf":
        if not hasattr(TF_IDF, "__len__"):
            raise TypeError("Add __len__ to TF_IDF: return len(self.vocab).")
        return TF_IDF(tokenizer, training_texts)
    return {"indicator": TokenIndicator, "bow": BagOfWords}[name](tokenizer)


def run_fold(args, pairs, label_names, tokenizer, representation, model_type, split, root):
    """Adapted from tutorial cells 149-152; score a separate outer fold."""
    from architectures import train_neural_network_model, load, predict_probs
    system = f"{representation}_{model_type}"
    folder = root / system / f"seed-{split['seed']}" / f"fold-{split['fold']}"
    folder.mkdir(parents=True)
    fit_pairs = [pairs[i] for i in split["fit"]]
    dev_pairs = [pairs[i] for i in split["dev"]]
    eval_pairs = [pairs[i] for i in split["outer_eval"]]
    set_seed(split["training_seed"])
    started = time.perf_counter()
    encoder = make_encoder(representation, tokenizer, [x for x, _ in fit_pairs])
    if representation == "tfidf":
        np.save(folder / "idf.npy", encoder.idf)
    prefix = str(folder / "model")
    history, best_epoch, _ = train_neural_network_model(
        training_set=fit_pairs, validation_set=dev_pairs,
        num_classes=len(label_names), encoder=encoder, save_to=prefix,
        model_type=model_type, hidden_dim=args.hidden_dim, dropout=args.dropout,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, num_epochs=args.epochs, device=args.device,
    )
    title = f"{system} / seed {split['seed']} / fold {split['fold']}"
    save_training_history(history, best_epoch, folder, title)
    model = load(prefix, device=args.device)
    probabilities = predict_probs([x for x, _ in eval_pairs], model, encoder,
                                  batch_size=args.batch_size)
    if (probabilities.shape != (len(eval_pairs), len(label_names))
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
        raise ValueError("Expected finite, normalized class probabilities.")
    predicted = probabilities.argmax(axis=1)
    gold = np.array([y for _, y in eval_pairs])
    scores, diagnostics = score_predictions(gold, predicted, label_names)
    save_predictions(folder / "predictions.csv", split["outer_eval"],
                     [x for x, _ in eval_pairs], gold, predicted, probabilities, label_names)
    save_diagnostics(diagnostics, folder, title)
    row = dict(system=system, representation=representation, model=model_type,
               seed=split["seed"], fold=split["fold"], **scores,
               best_epoch=int(best_epoch) + 1,
               parameters=sum(p.numel() for p in model.parameters()),
               fit_size=len(fit_pairs), dev_size=len(dev_pairs),
               eval_size=len(eval_pairs), seconds=time.perf_counter() - started)
    return row


def main():
    from datasets import load_dataset
    from tokenizers import Tokenizer

    # Collect experiment settings.
    args = SimpleNamespace(
        topic=TOPIC,
        output=OUTPUT_DIR,
        seeds=SEEDS,
        folds=NUM_FOLDS,
        inner_fraction=INNER_FRACTION,
        epochs=NUM_EPOCHS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        rope=ROPE,
        device=DEVICE,
    )

    # Create a new results directory.
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose a new OUTPUT_DIR to avoid mixing results.")
    root.mkdir(parents=True, exist_ok=True)

    # Load the selected topic and prepare text-label pairs.
    data = load_dataset("cardiffnlp/tweet_eval", args.topic, split="train")
    label_names = data.features["label"].names
    pairs = [(item["text"], int(item["label"])) for item in data]
    labels = np.array([label for _, label in pairs])

    # Use the same tokenizer and CV splits for every system.
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save(str(root / "tokenizer.json"))

    splits = make_splits(
        labels, args.seeds, args.folds, args.inner_fraction
    )

    # Save settings and splits for reproducibility.
    save_settings(root / "config.csv", {
        **vars(args),
        "label_names": label_names,
        "tokenizer": "bert-base-uncased",
        "primary_metric": "macro_f1",
        "checkpoint_selection": "minimum inner-validation NLL",
    })
    save_split_indices(root / "splits.csv", splits)

    # Save class counts to describe the dataset.
    save_csv(root / "class_counts.csv", [
        {
            "label": name,
            "count": int((labels == i).sum()),
        }
        for i, name in enumerate(label_names)
    ])

    # Run all representation-model combinations.
    rows = []
    total = len(splits) * len(REPRESENTATIONS) * len(MODELS)

    for representation, model_type in itertools.product(
        REPRESENTATIONS, MODELS
    ):
        for split in splits:
            print(
                f"\nRun {len(rows) + 1}/{total}: "
                f"{representation}/{model_type}, "
                f"seed={split['seed']}, fold={split['fold']}"
            )

            result = run_fold(
                args, pairs, label_names, tokenizer,
                representation, model_type, split, root,
            )
            rows.append(result)

            # Save progress after each completed run.
            save_csv(root / "fold_metrics.csv", rows)

    # Generate summary tables, plots and Bayesian comparisons.
    summarize(rows, root)
    compare_systems(rows, root, args.seeds, args.folds, args.rope)

    print(f"\nResults saved to {root.resolve()}")


if __name__ == "__main__":
    main()
