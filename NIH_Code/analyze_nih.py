# analyze_nih.py
"""
Multi-label NN Optimization on NIH Chest X-Ray Dataset

"""

# import
import time
import numpy as np
import torch
import torch.nn as nn
from models import (
    MultiLabelMobileNet, MultiLabelResNet
)
from nihdataset import NIHChestXRayDataset
import torchvision.transforms as transforms
from functions import (
    get_data_loaders,
    train_one_epoch,
    validate_one_epoch,
    fine_tune_bad_labels,
    give_epoch_stats,
    plot_roc_curve,
    plot_images_classification,
    find_best_thresholds_per_label
)

from performance_metrics import compute_fn_fp_rate, compute_weighted_f1
from torch.optim import Adam

#############################################
# 1) USER CONFIGURATION SECTION
#############################################

# -- DIRECTORIES --
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # parent of NIH_Code
data_dir = BASE_DIR/ "Data" / "archive"
cache_dir = BASE_DIR /"Cached_Data" / "cache_224_tot_90_jpg"
NIH_LABELS_CSV = BASE_DIR /"Information"/ "Data_Entry_2017.csv"

# -- SUBSET SETTINGS --
MAX_IMAGES = 50000  # Number of images to use (subset) to keep runtime manageable, this code
# chooses a representative subset, which is roughly of this size and aims to keep the proportions
# of different pathologies in the representative subset similar to the original distribution
# favoring thereby pathologies over negatives
NUM_LABELS = 14  # Use only this number of labels, choosing the most frequent in descending order

# -- EPOCHS SETTINGS --
NUM_EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = 2

initial_prob_threshold = 0.2
prob_thresholds = np.ones(NUM_LABELS) * initial_prob_threshold
thresholds_by_disease = True  # Optimize threshold per disease to improve f1 score on tuning set
derive_negatives = True # The Negative label is assigned if no disease probability exceeds its threshold.
f1_neg_threshold = 0.85
f1_pos_threshold = 0.30

delta_loss = 0.002

pretrained_model = 'MultiLabelMobileNet'
# pretrained_model = 'MultiLabelResNet'
train_full_model = True

preset_pos_weights = True

# NIH dataset has 14 disease labels:
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# --------------------
# MAIN
# --------------------
if __name__ == "__main__":

    """
        1) Instantiate multi-label classifier
        2) Load and split NIH Chest X-Ray data
        3) Train and validate each epoch in NUM_EPOCHS
        4) Fine-tune labels with high FN rate
        5) Compute and save per-disease statistics
        6) Plot and save ROC curves and sample predictions
    """
    start = time.perf_counter()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    #############################################
    # 1) Model
    #############################################
    if pretrained_model == 'MultiLabelMobileNet':
        my_model = MultiLabelMobileNet(NUM_LABELS)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    elif pretrained_model == 'MultiLabelResNet':
        my_model = MultiLabelResNet(NUM_LABELS)
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Unknown pretrained_model: {pretrained_model}")

    my_model.to(device)

    #############################################
    # 2) DATA PREPARATION
    #############################################

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    nih_data = NIHChestXRayDataset(main_dir=BASE_DIR, max_size=MAX_IMAGES,
                                   top_x=NUM_LABELS, transform=transform,
                                   cache_dir=cache_dir, use_cache=True)

    print("NIH Dataset created")
    print(os.path.join(BASE_DIR, "/Information/Data_Entry_2017.csv"))
    top_labels = nih_data.top_labels
    num_diseases = len(top_labels)

    print(f'Using top {num_diseases} disease labels, which are: {top_labels}.')

    train_loader, tune_loader, eval_loader = get_data_loaders(nih_data, BATCH_SIZE, NUM_WORKERS)
    # t0 = time.time()
    # images, labels = next(iter(train_loader))
    # print("First batch load sec:", round(time.time() - t0, 2))
    # print("Batch shapes:", images.shape, labels.shape)

    print("Dataset length:", len(nih_data))
    print("Train loader size:", len(train_loader.dataset))
    print("Val loader size:", len(eval_loader.dataset))
    print("Tune loader size:", len(tune_loader.dataset))

    #############################################
    # 3) Setup Loss & Optimizer
    #############################################
    train_indices = train_loader.dataset.indices

    labels = nih_data.data_df.loc[train_indices, top_labels].values

    print("Setting up nn optimizer.")
    if preset_pos_weights:
        pos = labels.sum(axis=0)
        N = labels.shape[0]
        pos = np.clip(pos, 1, None)
        pos_weight = (N - pos) / pos
        pos_weight = np.clip(pos_weight, 1.0, None)
        pos_weight = np.log(pos_weight) + 1
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    if train_full_model:
        lr = 1e-4
        weight_decay = 1e-4

        for param in my_model.parameters():
            param.requires_grad = True

        params = my_model.parameters()

    else:
        requires_grad = False
        lr = 1e-3
        weight_decay = 0.0
        if pretrained_model == "MultiLabelMobileNet":
            params = my_model.model.classifier.parameters()
        elif pretrained_model == "MultiLabelResNet":
            params = my_model.model.fc.parameters()

    optimizer = Adam(params,
                     lr=lr,
                     # betas=(0.9, 0.999),
                     # eps=1e-8,
                     weight_decay=weight_decay
                     )

    #############################################
    # 4) Training and Evaluation
    #############################################
    val_loss_min = 10

    for epoch in range(NUM_EPOCHS):
        print('━' * 60)
        print(f"{'━' * 15} Epoch: {epoch + 1} {'━' * 15}")
        # ━━━━━━━━━━━━━━━━━━ Training ━━━━━━━━━━━━━━━━━
        print(f"{'━' * 15} Training {'━' * 15}")
        my_model, train_loss_avg = train_one_epoch(my_model, criterion, optimizer, train_loader,
                                                   device)


        if epoch <= 3:

            # Probability Threshold tuning
            # =========================
            print(f"{'━' * 15} Optimize Probability Thresholds {'━' * 15}")
            t0 = time.time()

            if thresholds_by_disease:
                thr_tensor = torch.as_tensor(prob_thresholds, device=device,
                                             dtype=torch.float32).view(1, -1)
                all_predictions, all_probs, all_labels, val_loss_avg \
                    = validate_one_epoch(my_model, tune_loader, device, criterion, thr_tensor,
                                         derive_negatives)

                prob_thresholds = find_best_thresholds_per_label(
                    probs=all_probs,
                    labels=all_labels,
                    top_labels=top_labels,
                    n_grid=40
                )

                for label, thr in zip(top_labels, prob_thresholds):
                    print(f"{label}: {thr:.3f}")

        # Evaluation
        # =========================
        print(f"{'━' * 15} Evaluation {'━' * 15}")
        t0 = time.time()

        thr_tensor = torch.as_tensor(prob_thresholds, device=device, dtype=torch.float32).view(1,
                                                                                               -1)
        all_predictions, all_probs, all_labels, val_loss_avg \
            = validate_one_epoch(my_model, eval_loader, device, criterion, thr_tensor,
                                 derive_negatives)

        dt = time.time() - t0
        print(f"Test Validation sec:", round(dt, 2))
        if epoch >= 1:
            all_predictions_tune, _, all_labels_tune, _ \
                = validate_one_epoch(my_model, tune_loader, device, criterion, thr_tensor,
                                     derive_negatives)
            [fn_rate, fp_rate] = compute_fn_fp_rate(all_predictions_tune, all_labels_tune)
            pos_weight = abs(1.0 / (1.0 - 0.2 * fn_rate + 1e-6))
            pos_weight = pos_weight.to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # if max(fp_rate) < 0.11 and (max(fn_rate) > 0.5 or np.mean(fn_rate) > 0.3):
            #     pos_weight = 1.0 / (1.0 - 0.1 * fn_rate + 1e-6)
            #     pos_weight = pos_weight.to(device)
            #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # else:
            #     criterion = nn.BCEWithLogitsLoss()

        if epoch == NUM_EPOCHS - 1:
            # Finetuning
            # Fine tune the last layer only for those labels,
            # where fn / (fn + tp) = fn_rate >= 0.3,
            # =========================
            [fn_rate, fp_rate] = compute_fn_fp_rate(all_predictions, all_labels)

            # bad_labels should just choose the id of the disease with too high fn_rates
            bad_labels = np.where(fn_rate > 0.30)[0].tolist()
            if bad_labels:
                my_model = fine_tune_bad_labels(my_model, train_loader.dataset, bad_labels, device, n_epochs=2,
                                                lr=1e-4)

                all_predictions, all_probs, all_labels, val_loss_avg \
                    = validate_one_epoch(my_model, eval_loader, device, criterion, thr_tensor,
                                         derive_negatives)

                [fn_rate, fp_rate] = compute_fn_fp_rate(all_predictions, all_labels)

        # Statistics
        # =========================
        print(f"{'━' * 15} Statistics {'━' * 15}")

        stop_metric, f1_neg, f1_pos = compute_weighted_f1(predictions=all_predictions,
                                                          labels=all_labels,
                                                          top_labels=top_labels)

        early_stopping = False
        if f1_neg >= f1_neg_threshold and f1_pos > f1_pos_threshold:
            early_stopping = True

        if val_loss_avg > val_loss_min + delta_loss:
            early_stopping = True
        else:
            val_loss_min = min(val_loss_avg, val_loss_min)

        val_accuracy, eval_stats = give_epoch_stats(f1_neg, f1_pos,
                                                    NUM_EPOCHS,
                                                    epoch,
                                                    top_labels,
                                                    all_predictions,
                                                    all_labels,
                                                    train_loss_avg,
                                                    val_loss_avg,
                                                    early_stopping)

        if f1_neg >= f1_neg_threshold and f1_pos > f1_pos_threshold:
            print(f"Early stop at epoch {epoch + 1} because we reached "
                  f"threshold f1 neg {f1_neg} over {f1_neg_threshold} "
                  f"and f1 pos {f1_pos} over {f1_pos_threshold}.")
            break

        if val_loss_avg > val_loss_min + delta_loss:
            print(f"Early stop at epoch {epoch + 1} because this epochs loss {val_loss_avg} "
                  f"compared to the minimal loss in the previous epochs {val_loss_min}.")
            break

    end = time.perf_counter()

    print(f"Elapsed for train/eval: {end - start:.4f} seconds")
    #############################################
    # 6) Save Statistics and Plot
    #############################################
    print('Saving statistics')

    start1 = time.perf_counter()
    eval_dir = os.path.join(".", "Evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    eval_csv_path = os.path.join(eval_dir, "nih_eval_stats.csv")
    eval_stats.to_csv(eval_csv_path, index=False)
    print("Writing Statistics to \n", os.path.abspath(eval_csv_path))

    print('Plotting roc curves')
    plot_roc_curve(all_probs, all_labels, top_labels, eval_dir)

    print('Plotting image samples')
    plot_images_classification(my_model, eval_loader, device, top_labels, thr_tensor, mean, std,
                               eval_dir, derive_negatives)
    end1 = time.perf_counter()
    print(f"Elapsed for stats/pics: {end1 - start1:.1f} seconds")
