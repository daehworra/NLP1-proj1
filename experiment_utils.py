import csv
import itertools
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_splits(labels, seeds, folds, inner_fraction):
    """Adapted from tutorial cell 124; generate shared indices once."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if counts.min() < folds:
        raise ValueError("Every class needs at least as many samples as CV folds.")
    splits = []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (outer_train, outer_eval) in enumerate(splitter.split(labels, labels)):
            training_seed = int(np.random.SeedSequence([seed, fold]).generate_state(1)[0])
            fit, dev = train_test_split(
                outer_train, test_size=inner_fraction,
                random_state=training_seed, stratify=labels[outer_train],
            )
            if any(len(np.unique(labels[idx])) != len(classes)
                   for idx in (fit, dev, outer_eval)):
                raise ValueError("A split is missing a class; adjust folds/inner fraction.")
            splits.append(dict(seed=seed, fold=fold, training_seed=training_seed,
                               fit=fit.tolist(), dev=dev.tolist(),
                               outer_eval=outer_eval.tolist()))
    return splits

def score_predictions(gold, predicted, label_names):
    """Extend tutorial cell 136 with F1 and per-class diagnostics."""
    labels = list(range(len(label_names)))
    stance_labels = [label_names.index("against"), label_names.index("favor")]
    scores = {
        "macro_f1": float(f1_score(gold, predicted, labels=labels,
                                   average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predicted)),
        "stance_f1": float(f1_score(gold, predicted, labels=stance_labels,
                                    average="macro", zero_division=0)),
    }
    diagnostics = {
        "label_order": label_names,
        "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist(),
        "classification_report": classification_report(
            gold, predicted, labels=labels, target_names=label_names,
            output_dict=True, zero_division=0),
    }
    return scores, diagnostics

def summarize(rows, root):
    """Adapted from tutorial cells 140/153; SD is descriptive, not a CI."""
    systems = sorted({row["system"] for row in rows})
    summary = []
    for system in systems:
        group = [r for r in rows if r["system"] == system]
        item = {"system": system, "n_scores": len(group)}
        for metric in ("macro_f1", "accuracy", "stance_f1"):
            values = np.array([r[metric] for r in group])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1))
        item["parameters"] = group[0]["parameters"]
        summary.append(item)
    summary.sort(key=lambda r: r["macro_f1_mean"], reverse=True)
    save_csv(root / "summary.csv", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = [[r["macro_f1"] for r in rows if r["system"] == s] for s in systems]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(values)
    for position, scores in enumerate(values, 1):
        ax.scatter(np.full(len(scores), position), scores, s=15, alpha=0.6)
    ax.set_xticks(range(1, len(systems) + 1))
    ax.set_xticklabels(systems, rotation=35, ha="right")
    ax.set_ylabel("Outer-fold macro-F1")
    ax.set_title("Representation and architecture comparison")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(root / "macro_f1.png", dpi=180)
    plt.close(fig)
    save_summary_table(summary, root)
    return summary

def compare_systems(rows, root, seeds, folds, rope):
    """Adapted from tutorial cells 162-163; explicitly align paired scores.

    Use the public baycomp implementation instead of an unavailable part1
    wrapper. Runs denotes repeated CV runs, not the number of fold scores.
    Compare one factor at a time; treat these comparisons as exploratory.
    """
    from baycomp import CorrelatedTTest
    grouped = {}
    for row in rows:
        key = (row["seed"], row["fold"])
        group = grouped.setdefault(row["system"], {})
        if key in group:
            raise ValueError("Duplicate system/seed/fold result.")
        group[key] = row["macro_f1"]
        
    expected = sorted((seed, fold) for seed in seeds for fold in range(folds))
    
    for scores in grouped.values():
        if sorted(scores) != expected:
            raise ValueError("Bayesian comparison requires all paired folds.")
        
    comparisons = []
    metadata = {r["system"]: r for r in rows}
    
    for a, b in itertools.combinations(sorted(grouped), 2):
        if (metadata[a]["representation"] != metadata[b]["representation"]
                and metadata[a]["model"] != metadata[b]["model"]):
            continue
        x = np.array([grouped[a][key] for key in expected])
        y = np.array([grouped[b][key] for key in expected])
        pa, ptie, pb = CorrelatedTTest.probs(x, y, rope=rope, runs=len(seeds))
        comparisons.append(dict(system_a=a, system_b=b, metric="macro_f1",
                                rope=rope, mean_b_minus_a=float((y - x).mean()),
                                p_a_better=float(pa), p_equivalent=float(ptie),
                                p_b_better=float(pb)))
        
    save_csv(root / "bayesian_comparisons.csv", comparisons)

def save_settings(path, settings):
    """Flatten nested metadata into a human-readable key/value CSV."""
    rows = []
    def visit(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(f"{prefix}.{index}", item)
        else:
            rows.append({"setting": prefix, "value": value})
    visit("", settings)
    save_csv(path, rows)


def save_split_indices(path, splits):
    """Store shared splits once, including row order for reproducibility."""
    rows = []
    for split in splits:
        for subset in ("fit", "dev", "outer_eval"):
            for position, index in enumerate(split[subset]):
                rows.append(dict(seed=split["seed"], fold=split["fold"],
                                 training_seed=split["training_seed"],
                                 subset=subset, position=position, sample_index=index))
    save_csv(path, rows)


def save_training_history(history, best_epoch, folder, title):
    """Save one CSV and one two-panel figure for each training run.

    best_epoch is zero-based, matching train_neural_network_model.
    Training metrics are recorded during updates; validation metrics use eval mode.
    """
    folder = Path(folder)
    rows = [dict(epoch=i + 1, train_loss=r["train_metrics"]["loss"],
                 validation_loss=r["val_metrics"]["loss"],
                 train_accuracy=r["train_metrics"]["accuracy"],
                 validation_accuracy=r["val_metrics"]["accuracy"],
                 selected_checkpoint=(i == best_epoch))
            for i, r in enumerate(history)]
    save_csv(folder / "history.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(axes, ("loss", "accuracy"),
                                 ("Negative log-likelihood", "Accuracy")):
        for prefix, legend, color in (("train", "Training", "#147d92"),
                                       ("validation", "Inner validation", "#d8862b")):
            ax.plot([r["epoch"] for r in rows], [r[f"{prefix}_{metric}"] for r in rows],
                    label=legend, color=color)
        ax.axvline(best_epoch + 1, color="#6b7280", linestyle="--",
                   label=f"Selected epoch: {best_epoch + 1}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        if metric == "accuracy":
            ax.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(folder / "training_curves.png", dpi=180)
    plt.close(fig)


def save_diagnostics(diagnostics, folder, title):
    """Export per-class metrics and a labeled confusion matrix."""
    folder = Path(folder)
    names = diagnostics["label_order"]
    matrix = np.asarray(diagnostics["confusion_matrix"])
    save_csv(folder / "confusion_matrix.csv", [
        {"actual_label": label, **{f"predicted_{name}": int(matrix[i, j])
                                   for j, name in enumerate(names)}}
        for i, label in enumerate(names)])
    report = diagnostics["classification_report"]
    save_csv(folder / "classification_report.csv", [
        {"label": label, **report[label]} for label in names + ["macro avg", "weighted avg"]])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_predictions(path, indices, texts, gold, predicted, probabilities, label_names):
    """Keep predictions readable, including class names and probabilities."""
    rows = []
    for index, text, y, pred, probs in zip(indices, texts, gold, predicted, probabilities):
        rows.append(dict(sample_index=int(index), text=text,
                         gold_label=label_names[int(y)], predicted_label=label_names[int(pred)],
                         **{f"prob_{name}": float(prob) for name, prob in zip(label_names, probs)}))
    save_csv(path, rows)


def save_summary_table(summary, root):
    """Create a presentation-friendly image alongside full-precision summary.csv."""
    headers = ["System", "Macro-F1\nmean ± SD", "Accuracy\nmean ± SD",
               "Stance F1\nmean ± SD", "Parameters"]
    cells = [[r["system"],
              *[f"{float(r[m + '_mean']):.3f} ± {float(r[m + '_sd']):.3f}"
                for m in ("macro_f1", "accuracy", "stance_f1")],
              f"{int(r['parameters']):,}"] for r in summary]
    fig, ax = plt.subplots(figsize=(12, max(3, 1.8 + len(cells) * .45)))
    ax.axis("off")
    ax.set_title("Cross-validation results\nFold SD is descriptive, not a confidence interval",
                 fontsize=13, pad=18)
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=[.26, .20, .20, .20, .14])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce5eb")
        if row == 0:
            cell.set_facecolor("#173c50")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.3)
        else:
            cell.set_facecolor("#eef5f8" if row % 2 else "white")
    fig.tight_layout()
    fig.savefig(Path(root) / "summary_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add charts and CSV views to existing results.")
    parser.add_argument("--results", required=True, type=Path)
