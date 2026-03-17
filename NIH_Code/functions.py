"""
functions.py

Main utilities for:
  • loading and splitting the NIH Chest X-Ray dataset
  • training and validating a multi-label CNN
  • computing false-negative rates and per-label metrics
  • fine-tuning model heads on underperforming train_labels
  • aggregating epoch statistics
  • plotting ROC curves and sample classifications

Functions:
  get_data_loaders:
    Split a NIHChestXRayDataset into train/validation DataLoaders.

  train_one_epoch:
    Run one training pass over the train DataLoader, return updated model and average loss.

  validate_one_epoch:
    Evaluate model on validation DataLoader, return predictions, probabilities, train_labels, and average loss.

  fine_tune_bad_labels:
    Freeze backbone and fine-tune only classifier heads on samples whose train_labels exhibit high FN rates.

  epoch_stats:
    Update a pandas DataFrame with per-label confusion counts and performance metrics; print epoch summary.

  plot_roc_curve:
    Plot and save one combined figure with multi-label ROC curves using continuous scores.

  plot_images_classification:
    Display and save a grid of correctly and incorrectly classified chest X-rays with predicted vs. true train_labels.
"""
import os.path
import numpy as np
import time

import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from performance_metrics import give_eval_stats


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
                         n_epochs=2, lr=1e-4):
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
        return model

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

    if hasattr(model.model, 'fc'):
        if partial_unfreeze:
            for param in model.model.layer4.parameters():
                param.requires_grad = True
        head_params = list(model.model.fc.parameters())


    elif hasattr(model.model, 'classifier'):
        if partial_unfreeze:
            for param in model.model.features[-2:].parameters():
                param.requires_grad = True
        head_params = list(model.model.classifier.parameters())

    for param in head_params:
        param.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    weight_decay = 1e-4

    # 4) Build a fresh optimizer optimizer on all unfrozen params
    from torch.optim import Adam
    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)

    # 5) Fine-tune loop
    import torch.nn as nn
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
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(ft_loader) if len(ft_loader) > 0 else 0.0
        print(f"[Fine-tune {epoch + 1}/{n_epochs}] Loss: {avg_loss:.4f}")
    # 6) Return model to eval mode
    model.eval()
    return model


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
        print(f"{'━' * 15} Per-disease Stats with detail for final (Epoch {epoch + 1})"
              f" {'━' * 15}")
        print(eval_stats[["label", "total_pos",
                          "tp", "fp", "tn", "fn",
                          "f1_score", "auroc"]])
        print('━' * 60)

    return val_accuracy, eval_stats


def find_best_thresholds_per_label(probs: torch.Tensor,
                                   labels: torch.Tensor,
                                   n_grid: int = 40):
    """
    probs:  [N, L] float in [0,1]
    train_labels: [N, L] {0,1}
    returns: np.array thresholds [L]
    """
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)

    L = probs_np.shape[1]
    thresholds = np.zeros(L, dtype=np.float16)

    # search grid from 0.1..0.6
    grid = np.linspace(0.1, 0.6, n_grid)

    for j in range(L):

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

            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        thresholds[j] = best_thr

    return thresholds


def give_predictions(probs: torch.Tensor,
                     thr,
                     top_labels,
                     derive_negatives):
    """
    probs: [N, L]
    thresholds: array-like [L] or scalar
    returns: preds [N,L] long {0,1}
    """

    thr = thr.to(probs.device)
    preds = (probs > thr).long()

    if derive_negatives:
        neg_idx = list(top_labels).index("Negative")
        disease_cols = [i for i in range(preds.size(1)) if i != neg_idx]
        neg_pred = (preds[:, disease_cols].sum(dim=1) == 0).long()
        preds[:, neg_idx] = neg_pred

    return preds


def plot_roc_curve(probabilities, label_tensors, top_labels, eval_dir):
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
    plt_savepath = os.path.join(eval_dir, 'roc_curves_nih.png')
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
                               derive_negatives=True):
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
    plt_savepath = os.path.join(eval_dir, 'image_sample_nih.png')
    fig.savefig(plt_savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


import random


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
