"""
functions.py

Main utilities for:
  • splitting NIH Chest X-Ray dataset objects into train/validation DataLoaders
  • training and validating a multi-label CNN
  • computing predictions from model probabilities using thresholds
  • fine-tuning the classifier layer on underperforming labels
  • saving model checkpoints
  • plotting ROC curves and sample classifications

Functions:
  get_data_loaders:
    Split dataset objects into train/validation DataLoaders.

  train_one_epoch:
    Run one training pass over the train DataLoader, return updated model and average loss.

  validate_one_epoch:
    Evaluate model on validation DataLoader, return predictions, probabilities, labels, and average loss.

  give_predictions:
    Convert probability outputs into binary predictions using per-label thresholds
    and optional handling for the "Negative" class.

  fine_tune_bad_labels:
    Freeze the backbone and fine-tune only the classifier layer and optionally the last backbone block
     on samples whose labels exhibit high FN rates.

  find_best_thresholds_per_label:
    Calculate best thresholds per label inside a grid based on the F1 score.

  save_model:
    Save model, optimizer state, epoch, thresholds, and labels to disk.

  plot_roc_curve:
    Plot and save one combined figure with multi-label ROC curves using continuous scores.

  plot_images_classification:
    Display and save a grid of correctly and incorrectly classified chest X-rays with predicted vs. true labels.
"""
import os
import os.path
import numpy as np
import time

import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib
from torch.optim import Adam

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

from sklearn.metrics import precision_recall_curve, average_precision_score


def get_data_loaders(BATCH_SIZE, NUM_WORKERS,
                     nih_dataset,
                     pin_memory=False):
    prefetch_factor = 2 if NUM_WORKERS > 0 else None

    train_loader = DataLoader(
        Subset(nih_dataset, nih_dataset.train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=prefetch_factor,
    )

    tune_loader = DataLoader(
        Subset(nih_dataset, nih_dataset.tune_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=prefetch_factor,
    )

    test_loader = DataLoader(
        Subset(nih_dataset, nih_dataset.test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=prefetch_factor,
    )

    return train_loader, tune_loader, test_loader


def train_one_epoch(my_model, criterion, optimizer, train_loader, device,
                    scaler, use_amp,
                    verbose=False):
    my_model.train()
    running_loss = 0.0
    for i, (images, label_tensors, _) in enumerate(train_loader):
        if i == 0:
            t0 = time.time()
        images = images.to(device)
        label_tensors = label_tensors.to(device, dtype=torch.float32)

        # Backpropagation of gradient
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                outputs = my_model(images)
                loss = criterion(outputs, label_tensors)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = my_model(images)
            loss = criterion(outputs, label_tensors)
            loss.backward()
            optimizer.step()
        if verbose:
            dt = time.time() - t0
            if 0 <= i % 100 <= 10:
                print(f"Train step {i} sec:", round(dt, 2))
        # Total loss
        running_loss += loss.item()

    train_loss_avg = running_loss / len(train_loader)
    dt = time.time() - t0
    print(f"Train Time sec:", round(dt, 2))
    return my_model, train_loss_avg


def validate_one_epoch(my_model, test_loader, device,
                       criterion, thr_tensor, use_amp, derive_negatives=True):
    t0 = time.time()
    my_model.eval()
    all_probs, all_labels, all_patient_ids = [], [], []
    running_loss = 0.0

    base_dataset = test_loader.dataset.dataset if isinstance(test_loader.dataset,
                                                             Subset) else test_loader.dataset
    top_labels = base_dataset.top_labels

    with torch.no_grad():
        for images, label_tensors, patient_ids in test_loader:
            images = images.to(device)
            label_tensors = label_tensors.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = my_model(images)
                loss = criterion(outputs, label_tensors.float())
            probs = torch.sigmoid(outputs)
            running_loss += loss.item()

            all_probs.append(probs.cpu())
            all_labels.append(label_tensors.cpu())
            all_patient_ids.extend(patient_ids.cpu().tolist())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    val_loss_avg = running_loss / len(test_loader)

    all_predictions = give_predictions(probs=all_probs, thr=thr_tensor, top_labels=top_labels,
                                       derive_negatives=derive_negatives)
    dt = time.time() - t0
    print(f"Validation time (s):", round(dt, 2))

    return all_predictions, all_probs, all_labels, all_patient_ids, val_loss_avg


def fine_tune_bad_labels(model, train_subset, bad_label_ids, device,
                         partial_unfreeze=True,
                         pin_memory=False,
                         n_epochs=1,
                         lr_bb=1e-5,
                         lr_head=3e-5):
    """
    After an epoch, call this to fine-tune only on train_labels with
    false-negative rate > threshold_fn. Freezes most of the backbone,
    always updates the classifier head,
    and optionally unfreezes the last backbone block.
    """
    # 1) Identify all sample‐indices where at least one bad label is present.
    label_matrix = (
        train_subset.dataset
        .data_df
        .iloc[train_subset.indices][train_subset.dataset.top_labels]
        .values
    )

    # find rows where any of the bad train_labels is present
    bad_mask = (label_matrix[:, bad_label_ids] == 1).any(axis=1)
    bad_indices = np.flatnonzero(bad_mask).tolist()
    if len(bad_indices) == 0:
        print("No training samples found for bad train_labels. Skipping fine-tuning.")
        return model, None

    # 2) Create a Subset and DataLoader for just those samples
    subset = Subset(train_subset, bad_indices)
    ft_loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory
    )

    # 3) Freeze all model weights except the classifier head and optionally the last backbone block
    for param in model.parameters():
        param.requires_grad = False

    backbone = []
    if hasattr(model.model, 'fc'):
        if partial_unfreeze:
            backbone_layer = model.model.layer4
        head_layer = model.model.fc

    elif hasattr(model.model, 'classifier'):
        if partial_unfreeze:
            backbone_layer = model.model.features[-2:]
        head_layer = model.model.classifier[1]

    else:
        raise ValueError("Unsupported model head")

    if partial_unfreeze:
        backbone = list(backbone_layer.parameters())
        for p in backbone:
            p.requires_grad = True
    head = list(head_layer.parameters())

    for p in head:
        p.requires_grad = True

    good_label_ids = [i for i in range(head_layer.out_features) if i not in bad_label_ids]

    params = [p for p in model.parameters() if p.requires_grad]
    weight_decay = 1e-4

    # 4) Build a fresh optimizer optimizer on all unfrozen params
    if partial_unfreeze:
        optimizer = Adam(
            [
                {"params": backbone, "lr": lr_bb},
                {"params": head, "lr": lr_head},
            ],
            weight_decay=weight_decay
        )
    else:
        optimizer = Adam(
            [
                {"params": head, "lr": lr_head},
            ],
            weight_decay=weight_decay
        )

    # 5) Fine-tune loop
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for images, labels, _ in ft_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(images)
            # compute loss only on bad train_labels
            loss = criterion(
                logits[:, bad_label_ids],
                labels[:, bad_label_ids]
            )
            loss.backward()

            with torch.no_grad():
                if head_layer.weight.grad is not None:
                    head_layer.weight.grad[good_label_ids] = 0
                if head_layer.bias is not None and head_layer.bias.grad is not None:
                    head_layer.bias.grad[good_label_ids] = 0
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(ft_loader) if len(ft_loader) > 0 else 0.0
        print(f"[Fine-tune {epoch + 1}/{n_epochs}] Loss: {avg_loss:.4f}")
    # 6) Return model to eval mode
    model.eval()
    return model, optimizer


def find_best_thresholds_per_label(probs: torch.Tensor,
                                   labels: torch.Tensor,
                                   min_thr=0.15,
                                   max_thr=0.85,
                                   n_grid: int = 80):
    """
    probs:  [N, L] float in [0,1]
    train_labels: [N, L] {0,1}
    returns: np.array thresholds [L]
    """
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)

    L = probs_np.shape[1]
    thresholds = np.zeros(L, dtype=np.float16)

    for j in range(L):

        # search grid from min_thr to max_thr. Negatives should not get too low threshold
        if j == 0:
            grid = np.linspace(0.5, max_thr, n_grid)
        else:
            grid = np.linspace(min_thr, max_thr, n_grid)

        y_true = labels_np[:, j]
        p = probs_np[:, j]

        best_thr = 0.5
        best_f1 = -1.0

        # compute F1 for each threshold
        for thr in grid:
            y_pred = (p > thr).astype(int)

            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            # precision/recall safe
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)

            # optimize threshold based on f1
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        thresholds[j] = best_thr

    return thresholds


def give_predictions(probs: torch.Tensor,
                     thr: torch.Tensor,
                     top_labels,
                     derive_negatives: bool) -> torch.Tensor:
    """
    Convert probabilities into binary predictions.

    Args:
        probs: Tensor of shape [N, L] with probabilities in [0, 1].
        thr: Threshold tensor of shape [1, L], [L], or scalar.
        top_labels: Sequence of label names, including "Negative".
        derive_negatives: If True, derive the "Negative" label using a
            competition rule against disease probabilities.

    Returns:
        preds: Tensor of shape [N, L] with binary predictions {0, 1}.
    """
    thr = torch.as_tensor(thr, device=probs.device, dtype=probs.dtype)

    if thr.ndim == 0:
        thr = thr.view(1, 1)
    elif thr.ndim == 1:
        thr = thr.view(1, -1)

    preds = (probs > thr).long()

    neg_idx = list(top_labels).index("Negative")
    disease_cols = [i for i in range(probs.size(1)) if i != neg_idx]

    no_disease_over_threshold = preds[:, disease_cols].sum(dim=1) == 0

    if derive_negatives:

        neg_prob = probs[:, neg_idx]
        neg_thr = thr[0, neg_idx]

        disease_probs = probs[:, disease_cols]
        max_disease_prob = disease_probs.max(dim=1).values

        k = min(2, disease_probs.size(1))
        topk_vals, _ = disease_probs.topk(k=k, dim=1)
        top2_disease_sum = topk_vals.sum(dim=1)

        neg_pred = (
                (neg_prob >= neg_thr) &
                (neg_prob > max_disease_prob) &
                (top2_disease_sum < (neg_prob + 0.05))
        ).long()

        preds[:, neg_idx] = neg_pred
        # If no disease predicted set negative to avoid all labels = 0 case
        preds[no_disease_over_threshold,neg_idx] = 1

        # If Negative is predicted, force all disease labels to 0.
        mask_neg = (neg_pred == 1)
        if mask_neg.any():
            preds[mask_neg] = 0
            preds[mask_neg, neg_idx] = 1
    else:
        # Keep predictions logically consistent also in the non-derived case:
        # if any disease is predicted, force Negative to 0.
        has_disease = preds[:, disease_cols].sum(dim=1) > 0
        preds[has_disease, neg_idx] = 0

    return preds


def save_model(model, optimizer, epoch, prob_thresholds, top_labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "prob_thresholds": prob_thresholds,
        "top_labels": list(top_labels),
    }, path)

    print(f"Model saved to {path}")


def plot_roc_curve(probabilities, label_tensors, top_labels, eval_dir, name=''):
    """
    Plot multi-label ROC curves and saves them.
    """

    fig, ax = plt.subplots(figsize=(9, 9))

    for idx, disease_name in enumerate(top_labels):
        # keep the *raw* probability scores (no .astype(int))
        prob_np = probabilities[:, idx].detach().cpu().numpy()
        true_np = label_tensors[:, idx].detach().cpu().numpy()

        # compute false/true positive rates and area under curve
        if np.unique(true_np).size < 2:
            continue
        fpr, tpr, _ = roc_curve(true_np, prob_np)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            label=f'{disease_name} (AUC: {roc_auc:0.2f})'
        )

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for NIH Diseases')
    ax.legend(loc='lower right')
    ax.grid(True)

    # show once and save once
    plt_savepath = os.path.join(eval_dir, f"roc_curves_nih{name}.png")
    fig.savefig(plt_savepath)
    print("Saving Roc Curves to \n", os.path.abspath(plt_savepath))
    plt.close(fig)


def plot_images_classification(model,
                               test_loader,
                               device,
                               top_labels,
                               thr_tensor,
                               mean,
                               std,
                               eval_dir,
                               derive_negatives=True,
                               name=''):
    # Reverse normalization
    inv_normalize = transforms.Normalize(
        mean=- mean / std,
        std=1 / std
    )

    # Take one batch and evaluate
    model.eval()
    images, labels, patient_ids = next(iter(test_loader))
    images, labels, patient_ids = images.to(device), labels.to(device), patient_ids.to(device)

    with torch.no_grad():
        logits = model(images)

    probs = torch.sigmoid(logits)
    preds = give_predictions(probs, thr_tensor, top_labels, derive_negatives)

    pred_np = preds.cpu().numpy()
    label_np = labels.cpu().numpy()
    # n = number of samples, k = number of train_labels
    n, k = pred_np.shape
    assert k == len(top_labels)

    # Assign to each sample according to its label one of the top train_labels,
    # depending on which label is the first one to have a 1.
    # We start with the first of the top train_labels(negative) and search all train_labels in the sample,
    # which have a 1 at position 0 (negative). We save all those indices of the train_labels
    # and consider the remaining samples. Then we move on with the first disease label
    # and search those train_labels which have a one in the position 1 (first disease label).
    # We remove all those and continue with disease 2.

    assignment = []
    unassigned = np.ones(n, dtype=bool)

    for i in range(k):
        # all original indices with this label *and* still unassigned
        idxs = np.where((label_np[:, i] == 1) & unassigned)[0]
        assignment.append(idxs)
        unassigned[idxs] = False

        # only keep train_labels that actually occurred
    assignment = [idxs for idxs in assignment if idxs.size > 0]

    # Height corresponds to the number of diseases + 1 (for negatives),
    # which have at least one label assigned to
    grid_height = len(assignment)

    # For each label that occurs in the selected batch
    grid_length = np.zeros(grid_height, dtype=int)

    sample_idx = np.array([], dtype=int)
    for i in range(grid_height):
        # Take all train_labels where i-th is positive
        idxs = assignment[i]  # e.g. array([2,7,13,...])
        label_rows = label_np[idxs, :]  # shape (num_pos_for_class_i,)
        pred_rows = pred_np[idxs, :]  # same shape

        # 2. get local “mismatch” positions
        local_mismatches = (label_rows != pred_rows).any(axis=1)
        local_matches = (label_rows == pred_rows).all(axis=1)

        # 3. if you need the *original* sample indices, map back:
        mismatches = idxs[local_mismatches]
        matches = idxs[local_matches]
        n_mism = min(len(mismatches), 2)
        n_match = min(len(matches), 3 - n_mism, 2)
        grid_length[i] = n_mism + n_match

        sample_idx = np.concatenate(
            [sample_idx, mismatches[:n_mism], matches[:n_match]]
        )
    max_grid_length = max(grid_length)
    assert max_grid_length <= 3

    # ---- plot ----
    # Create a grid of subplots: rows = grid_height, cols = max_grid_length
    fig, axes = plt.subplots(
        nrows=grid_height,
        ncols=max_grid_length,
        figsize=(max_grid_length * 2, grid_height * 2),  # 2″ per image cell
        dpi=300
    )
    axes = np.atleast_1d(axes).ravel()

    # Fill in each used cell
    for cell_idx, img_idx in enumerate(sample_idx):
        ax = axes[cell_idx]

        # Reconstruct and denormalize image
        img_tensor = inv_normalize(images[img_idx])
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        img = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Show image
        ax.imshow(img, interpolation='nearest', aspect='equal')

        # Build label strings
        pred_ids = np.where(pred_np[img_idx] == 1)[0]
        true_ids = np.where(label_np[img_idx] == 1)[0]
        pred_str = f"pred: {', '.join(top_labels[j] for j in pred_ids) or '∅'}"
        true_str = f"true: {', '.join(top_labels[j] for j in true_ids) or '∅'}"

        # Draw text with a readable font size
        ax.text(
            0, 1.2, pred_str,
            transform=ax.transAxes,
            fontsize=8,
            color='blue',
            ha='left',
            clip_on=False
        )
        ax.text(
            0, 1.1, true_str,
            transform=ax.transAxes,
            fontsize=8,
            color='red',
            ha='left',
            clip_on=False
        )

        ax.axis("off")

    # Turn off any extra axes (when grid isn't completely filled)
    for ax in axes[len(sample_idx):]:
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt_savepath = os.path.join(eval_dir, f"image_sample_nih{name}.png")
    fig.savefig(plt_savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_precision_recall_curves(y_true, y_prob, label_names, eval_dir=None, name = ''):
    """
    Plot Precision-Recall curves for all labels in a single figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, label in enumerate(label_names):
        y_t = y_true[:, i]
        y_p = y_prob[:, i]

        if np.sum(y_t) == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_t, y_p)
        ap = average_precision_score(y_t, y_p)

        ax.plot(recall, precision, label=f"{label} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt_savepath = os.path.join(eval_dir, f"pc_nih{name}.png")
    fig.savefig(plt_savepath)
    print("Saving PC Curves to \n", os.path.abspath(plt_savepath))
    plt.close(fig)

    return fig


def plot_probability_distributions(y_true, y_prob, label_names, bins=40, eval_dir=None, name = ''):
    """
    Plot probability distributions (pos vs neg) for all labels in a grid.
    """
    n_labels = len(label_names)
    n_cols = int(np.ceil(np.sqrt(n_labels)))
    n_rows = int(np.ceil(n_labels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    axes = np.array(axes).reshape(-1)

    for i, label in enumerate(label_names):
        ax = axes[i]

        y_t = y_true[:, i]
        y_p = y_prob[:, i]

        pos_scores = y_p[y_t == 1]
        neg_scores = y_p[y_t == 0]

        ax.hist(neg_scores, bins=bins, alpha=0.6, label="Neg", density=True)
        ax.hist(pos_scores, bins=bins, alpha=0.6, label="Pos", density=True)

        ax.set_title(label)
        ax.set_xlabel("Prob")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Shared legend (cleaner)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle("Probability Distributions (Pos vs Neg)", fontsize=14)
    plt.tight_layout()

    plt_savepath = os.path.join(eval_dir, f"prob_distr_nih{name}.png")
    fig.savefig(plt_savepath)
    print("Saving Prob Distr Curves to \n", os.path.abspath(plt_savepath))
    plt.close(fig)
    return fig


def save_and_plot(results, eval_dir, top_labels, my_model, test_loader, device,
                  prob_thresholds, mean, std, derive_negatives, name=''):
    thr_tensor = (torch.as_tensor(prob_thresholds, device=device, dtype=torch.float32).
                  view(1, -1))
    eval_csv_path = os.path.join(eval_dir, f"nih_eval_stats{name}.csv")
    results["eval_stats"].to_csv(eval_csv_path, index=False)
    print("Writing Statistics to \n", os.path.abspath(eval_csv_path))

    all_probs = results["all_probs"]
    all_labels = results["all_labels"]
    print('Plotting roc curves')
    plot_roc_curve(all_probs, all_labels, top_labels, eval_dir, name)

    y_true = all_labels.detach().cpu().numpy().astype(int)
    y_prob = all_probs.detach().cpu().numpy()

    plot_precision_recall_curves(
        y_true=y_true,
        y_prob=y_prob,
        label_names=top_labels,
        eval_dir = eval_dir,
        name = name,
    )

    plot_probability_distributions(
        y_true=y_true,
        y_prob=y_prob,
        label_names=top_labels,
        eval_dir=eval_dir,
        name=name,
    )

    print('Plotting image samples')
    plot_images_classification(my_model, test_loader, device, top_labels, thr_tensor, mean, std,
                               eval_dir, derive_negatives, name)




import random

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

