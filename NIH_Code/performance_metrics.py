"""
performance_metrics.py

Compute and aggregate binary classification performance metrics.

Functions:
  compute_tp_fp_tn_fn:
    From prediction and label tensors, count TP, FP, TN, and FN per label.

  compute_weighted_f1:
    Compute a combined metric as 50/50 between:
      • F1 score of the negative class
      • weighted F1 score of all other labels (weighted by support).

  compute_fn_fp_rate:
    Compute false-negative rate and false-positive rate per label.

  give_performance_metrics:
    Compute accuracy, precision, recall, and F1 score from raw counts
    of true positives, true negatives, false positives, and false negatives.

  give_eval_stats:
    Populate a pandas DataFrame with per-label confusion counts and
    derived performance metrics (including AUROC when probabilities are provided).

  give_epoch_stats:
    Aggregate and print per-epoch statistics including losses, F1 scores,
    accuracy, AUROC, and per-label summaries.

  evaluate_and_report:
    Run full evaluation on a DataLoader:
      • forward pass via validate_one_epoch
      • compute FN/FP rates and weighted F1 metrics
      • optionally compute and print detailed evaluation statistics
      • return all predictions, probabilities, labels, and metrics
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from functions import validate_one_epoch


def compute_tp_fp_tn_fn(predictions, labels):
    predictions = predictions.int()
    labels = labels.int()

    tp = ((predictions == 1) & (labels == 1)).sum(dim=0).float()
    fp = ((predictions == 1) & (labels == 0)).sum(dim=0).float()
    fn = ((predictions == 0) & (labels == 1)).sum(dim=0).float()
    tn = ((predictions == 0) & (labels == 0)).sum(dim=0).float()

    return tp, fp, tn, fn


def compute_weighted_f1(predictions, labels, top_labels, negative_label="Negative", eps=1e-8):
    tp_all, fp_all, tn_all, fn_all = compute_tp_fp_tn_fn(predictions, labels)

    # negative goes separately
    neg_idx = list(top_labels).index(negative_label)

    f1s = []
    supports = []  # weights

    for j, _label in enumerate(top_labels):
        tp_j = tp_all[j].item()
        fp_j = fp_all[j].item()
        tn_j = tn_all[j].item()
        fn_j = fn_all[j].item()

        _acc, _prec, _rec, f1_j = give_performance_metrics(tp_j, tn_j, fp_j, fn_j)
        f1s.append(float(f1_j))

        support_j = tp_j + fn_j  # weight = real positives per class
        supports.append(int(support_j))

    # F1 of negatives
    f1_neg = f1s[neg_idx]

    # Weighted F1 of pathologies
    pos_idxs = [i for i in range(len(top_labels)) if i != neg_idx]
    pos_support_sum = sum(supports[i] for i in pos_idxs)

    if pos_support_sum <= 0:
        f1_pos = 0.0
    else:
        f1_pos = sum(f1s[i] * (supports[i] / (pos_support_sum + eps)) for i in pos_idxs)

    # Final metrics 50/50 f1 neg / weighted f1 pathologies
    stop_metric = 0.5 * f1_neg + 0.5 * f1_pos
    return stop_metric, f1_neg, f1_pos


def compute_fn_fp_rate(predictions, labels):
    tp, fp, tn, fn = compute_tp_fp_tn_fn(predictions, labels)
    fn_rate = torch.where((fn + tp) > 0, fn / (fn + tp), torch.zeros_like((fn + tp)))
    fp_rate = torch.where((fp + tn) > 0, fp / (fp + tn), torch.zeros_like((fp + tn)))

    return [fn_rate, fp_rate]


def give_performance_metrics(tp, tn, fp, fn):
    """
       Given scalar counts TP, TN, FP, FN, return:
         accuracy   = (TP + TN) / (TP + TN + FP + FN)
         precision  = TP / (TP + FP)            if TP + FP > 0 else 0
         recall     = TP / (TP + FN)            if TP + FN > 0 else 0
         f1_score   = 2·(precision·recall)/(precision+recall) if precision+recall > 0 else 0
       """

    total = (tp + tn + fp + fn)
    correct = (tp + tn)
    accuracy = correct / total
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return accuracy, precision, recall, f1_score


def give_eval_stats(epoch, top_labels, predictions, labels, probs=None):
    """
    Build a pandas DataFrame of per-label stats for logging or analysis.

    Arguments:
      epoch        (int):     training epoch number
      top_labels (List[str]): ordered list of label names; index 0 must be the 'negative' / 'no-finding' label
      predictions  (Tensor):  shape (N, K), raw binary predictions (0/1)
      labels       (Tensor):  shape (N, K), ground-truth binary train_labels (0/1)
      probs       (Tensor):  shape (N, K), probabilities in [0,1].
      If provided, AUROC is computed per label when possible.

    Returns:
      eval_stats (pd.DataFrame) with columns:
        - epoch
        - label
        - [for j=0 only] tp->tn, fp->fn
    """
    # 1) get raw counts for all K train_labels at once
    tp_all, fp_all, tn_all, fn_all = compute_tp_fp_tn_fn(predictions, labels)

    records = []
    # neg_idx = list(top_labels).index("Negative")
    records = []
    for j, label in enumerate(top_labels):
        tn_j = tn_all[j].item()
        fn_j = fn_all[j].item()
        tp_j = tp_all[j].item()
        fp_j = fp_all[j].item()
        total_pos = tp_j + fn_j
        # compute accuracy, precision, recall, f1
        acc_j, prec_j, rec_j, f1_j = give_performance_metrics(tp_j, tn_j, fp_j, fn_j)
        auc_j = np.nan
        if probs is not None:
            y_true = labels[:, j].cpu().numpy().astype(int)
            y_score = probs[:, j].cpu().numpy()
            if np.unique(y_true).size == 2:
                auc_j = roc_auc_score(y_true, y_score)
        # if j == neg_idx:
        #     # We say for the negative class that negative means 'Negative' = 1,
        #     # thus we invert the notation p and n for this class. This hasn't any
        #     # impact on the derived metrics
        #     tp_j_help = tn_j
        #     tn_j = tp_j
        #     tp_j = tp_j_help
        #     fp_j_help = fn_j
        #     fn_j = fp_j
        #     fp_j = fp_j_help
        records.append({
            "epoch": epoch + 1,
            "label": label,
            "total_pos": total_pos,
            "tp": tp_j,
            "fp": fp_j,
            "tn": tn_j,
            "fn": fn_j,
            "accuracy": round(acc_j, 3),
            "precision": round(prec_j, 3),
            "recall": round(rec_j, 3),
            "f1_score": round(f1_j, 3),
            "auroc": round(float(auc_j), 3) if not np.isnan(auc_j) else np.nan,
        })

    eval_stats = pd.DataFrame.from_records(
        records,
        columns=[
            "epoch", "label", "total_pos",
            "tp", "fp", "tn", "fn",
            "accuracy", "precision", "recall",
            "f1_score", "auroc"
        ]
    )

    return eval_stats


def give_epoch_stats(f1_neg,
                     f1_pos,
                     num_epochs,
                     epoch,
                     top_labels,
                     predictions,
                     label_tensors,
                     train_loss_avg,
                     val_loss_avg,
                     probs=None,
                     final_evaluation=False):
    eval_stats = give_eval_stats(epoch, top_labels, predictions, label_tensors, probs)
    val_f1_score = round(eval_stats['f1_score'].mean(), 2)  # in [0..1]
    val_accuracy = round(eval_stats['accuracy'].mean(), 2)  # in [0..1]
    val_auroc = round(eval_stats['auroc'].mean(), 2)  # in [0..1]
    # --------------------------------
    # Print Statistics
    # --------------------------------

    # Main summary of this epoch
    print('━' * 60)
    print(f"{'━' * 15} Epoch Evaluation {'━' * 15}")
    print(
        f"Epoch:{epoch + 1}, "
        f"Train Loss:{train_loss_avg:.2f}, "
        f"Val Loss:{val_loss_avg:.2f}, "
        f"Val F1 abs:{val_f1_score * 100:.2f}%, ",
        f"Val F1 neg:{f1_neg * 100:.2f}%, ",
        f"Val F1 pos:{f1_pos * 100:.2f}%, "
        f"Val Accuracy:{val_accuracy * 100:.2f}%, "
        f"Val Auroc:{val_auroc * 100:.2f}%, "
    )
    print(f"{'━' * 15} Per-disease Stats (Epoch {epoch + 1}) {'━' * 15}")
    print(eval_stats[["label", "total_pos", "precision", "recall", "f1_score", "auroc"]])
    print('━' * 60)

    # Per-disease stats at final epoch only
    if epoch == num_epochs - 1 or final_evaluation:
        pd.set_option("display.max_columns", None)
        print('━' * 60)
        print(f"{'━' * 15} Detailed Per-disease Stats (Epoch {epoch + 1})"
              f" {'━' * 15}")
        print(eval_stats[["label", "total_pos",
                          "tp", "fp", "tn", "fn",
                          "f1_score", "auroc"]])
        print('━' * 60)

    return val_accuracy, eval_stats


def evaluate_and_report(
        model,
        loader,
        device,
        criterion,
        prob_thresholds,
        use_amp,
        derive_negatives,
        top_labels,
        NUM_EPOCHS,
        epoch,
        train_loss_avg,
        give_stats=False,
        final_evaluation=False,
        title=None
):
    if title:
        print(f"{'━' * 15} {title}, Epoch {epoch} {'━' * 15}")

    thr_tensor = (torch.as_tensor(prob_thresholds, device=device, dtype=torch.float32).
                  view(1, -1))

    all_predictions, all_probs, all_labels, all_patient_ids, val_loss_avg = validate_one_epoch(
        model, loader, device, criterion, thr_tensor, use_amp, derive_negatives
    )

    fn_rate, fp_rate = compute_fn_fp_rate(all_predictions, all_labels)

    _, f1_neg, f1_pos = compute_weighted_f1(
        predictions=all_predictions,
        labels=all_labels,
        top_labels=top_labels
    )

    val_accuracy = None
    eval_stats = None
    if give_stats:
        val_accuracy, eval_stats = give_epoch_stats(
            f1_neg, f1_pos, NUM_EPOCHS, epoch,
            top_labels, all_predictions, all_labels,
            train_loss_avg, val_loss_avg, all_probs,
            final_evaluation=final_evaluation
        )

    return {
        "all_predictions": all_predictions,
        "all_probs": all_probs,
        "all_labels": all_labels,
        "patient_ids": all_patient_ids,
        "val_loss_avg": val_loss_avg,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
        "f1_neg": f1_neg,
        "f1_pos": f1_pos,
        "val_accuracy": val_accuracy,
        "eval_stats": eval_stats,
    }
