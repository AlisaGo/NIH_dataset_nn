# analyze_nih.py
"""
Multi-label NN Optimization on NIH Chest X-Ray Dataset

NIH Chest X-Ray Dataset:
Original resolution: 1024x1024
Scans: 112120
Number of patients: 30805
Mean images per patient: 3.64
Diseases by frequency
Negative              60361
Infiltration          19894
Effusion              13317
Atelectasis           11559
Nodule                 6331
Mass                   5782
Pneumothorax           5302
Consolidation          4667
Pleural_Thickening     3385
Cardiomegaly           2776
Emphysema              2516
Edema                  2303
Fibrosis               1686
Pneumonia              1431
Hernia                  227
dtype: int64
Patients by number of labels that change across their images
0     21341
1       181
2      3221
3      1892
4      1293
5       987
6       620
7       487
8       324
9       201
10      136
11       83
12       28
13        7
14        4
"""

# import
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from models import (
    MultiLabelMobileNet, MultiLabelResNet
)
from nihdataset import NIHChestXRayDataset, XRayStandardize
import torchvision.transforms as transforms
from functions import (
    get_data_loaders,
    train_one_epoch,
    validate_one_epoch,
    fine_tune_bad_labels,
    find_best_thresholds_per_label,
    set_global_seed,
    save_model,
    save_and_plot
)

from performance_metrics import evaluate_and_report
from torch.optim import Adam

#############################################
# 1) USER CONFIGURATION SECTION
#############################################

# -- DIRECTORIES --
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # parent of NIH_Code
data_dir = BASE_DIR / "Data" / "archive"
cache_dir = BASE_DIR / "Cached_Data" / "cache_224_tot_90_jpg"
NIH_LABELS_CSV = BASE_DIR / "Information" / "Data_Entry_2017.csv"

# -- Random seed --
SEED = 42
set_global_seed(SEED)

# -- Rescale images to speed up training --
do_rescale = True

# -- Used Cached Images with reduced size --
use_cache = True
if use_cache:
    do_rescale = False  # Cached images are already rescaled

# -- Used official train eval split --
USE_OFFICIAL_SPLIT = True
TRAIN_VAL_LIST = BASE_DIR / "Information" / "train_val_list.txt"
TEST_LIST = BASE_DIR / "Information" / "test_list.txt"

# # -- Exclude Labels --
EXCLUDED_LABELS = {
    "Nodule",  # Small-object difficulty, exclude due to resolution 1024 -> 320 or 224
    "Mass",  # difficult to detect, due to large within-class appearance variation
    "Hernia",  # Only 227 examples are too few!
}

# -- Partiel unfreeze bad labels --
partial_unfreeze_bad_labels = True  # Decides if to unfreeze just the head or also the
# last backbone block of the model for fine-tuning on less performant labels

# -- SUBSET SETTINGS --
MAX_IMAGES = 20000  # Number of images to use (subset) to keep runtime manageable, this code
# chooses a representative subset, which is roughly of this size and aims to keep the proportions
# of different pathologies in the representative subset similar to the original distribution
# favoring thereby pathologies over negatives
eval_frac = 0.15  # Wrt. max_images
NUM_LABELS_ALL = 14
NUM_LABELS = 11  # Use only this number of abels, choosing the most frequent in descending
# order

NUM_LABELS = min(NUM_LABELS_ALL - len(EXCLUDED_LABELS), NUM_LABELS)

# Prob thresholds for prediction
THRESHOLD_TUNE_EPOCH = 1
initial_prob_threshold = 0.25
prob_thresholds = np.ones(NUM_LABELS) * initial_prob_threshold
thresholds_by_disease = True  # Optimize threshold per disease to improve f1 score on tuning set
derive_negatives = True  # The Negative label is assigned if no disease probability exceeds its threshold.

# -- EPOCHS SETTINGS --
NUM_EPOCHS = 3
BATCH_SIZE = 32
NUM_WORKERS = 2

# -- Stopping criteria  --
f1_neg_threshold = 0.85
f1_pos_threshold = 0.4
delta_loss = 0.002

# -- Model --
# pretrained_model = 'MultiLabelMobileNet'
pretrained_model = 'MultiLabelResNet'
train_full_model = True

# -- Loss function weights --
# Preset and update pos. weights of the loss function
preset_pos_weights = False
update_pos_weights = False

# -- NIH dataset has 14 disease train_labels --
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
        4) Fine-tune train_labels with high FN rate
        5) Compute and save per-disease statistics
        6) Plot and save ROC curves and sample predictions
    """
    start = time.perf_counter()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        pin_memory = True
        NUM_WORKERS = 4
    else:
        pin_memory = False

    print("Using device:", device)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Unknown pretrained_model: {pretrained_model}")

    print(f"pretrained model:", pretrained_model)
    my_model.to(device)

    #############################################
    # 2) DATA PREPARATION
    #############################################
    if use_cache:
        IMG_SIZE = 224
    else:
        IMG_SIZE = 1024

    train_transform = transforms.Compose([
        XRayStandardize(do_rescale=do_rescale, size=IMG_SIZE, clip_percentiles=(1, 99)),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        XRayStandardize(do_rescale=do_rescale, size=IMG_SIZE, clip_percentiles=(1, 99)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    nih_data = NIHChestXRayDataset(main_dir=BASE_DIR, label_file=NIH_LABELS_CSV,
                                   use_official_split=USE_OFFICIAL_SPLIT,
                                   train_val_list_path=TRAIN_VAL_LIST, test_list_path=TEST_LIST,
                                   max_size=MAX_IMAGES, eval_frac=eval_frac, top_x=NUM_LABELS,
                                   train_transform=train_transform, eval_transform=eval_transform,
                                   excluded_labels=EXCLUDED_LABELS, cache_dir=cache_dir,
                                   random_seed=SEED, use_cache=use_cache)

    print("NIH Dataset created")
    top_labels = nih_data.top_labels
    num_diseases = len(top_labels)

    print(f'Using top {num_diseases} disease train_labels, which are: {top_labels}.')

    train_loader, tune_loader, test_loader = get_data_loaders(
        BATCH_SIZE,
        NUM_WORKERS,
        nih_dataset=nih_data,
        pin_memory=pin_memory,
    )

    # t0 = time.time()
    # images, train_labels = next(iter(train_loader))
    # print("First batch load sec:", round(time.time() - t0, 2))
    # print("Batch shapes:", images.shape, train_labels.shape)

    print("Dataset length:", len(nih_data))
    print("Train loader size:", len(train_loader.dataset))
    print("Test loader size:", len(test_loader.dataset))
    print("Tune loader size:", len(tune_loader.dataset))

    #############################################
    # 3) Setup Loss & Optimizer
    #############################################
    train_labels = nih_data.data_df.iloc[nih_data.train_idx][top_labels].values

    print("Setting up nn optimizer.")
    if preset_pos_weights:
        pos = train_labels.sum(axis=0)
        N = train_labels.shape[0]
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
    val_loss_min = float("inf")
    early_stopping = False

    for epoch in range(NUM_EPOCHS):
        print('━' * 60)
        print(f"{'━' * 15} Epoch: {epoch + 1} {'━' * 15}")
        # ━━━━━━━━━━━━━━━━━━ Training ━━━━━━━━━━━━━━━━━
        print(f"{'━' * 15} Training {'━' * 15}")
        my_model, train_loss_avg = train_one_epoch(my_model, criterion, optimizer, train_loader,
                                                   device, scaler, use_amp)

        if epoch == THRESHOLD_TUNE_EPOCH:

            # Probability Threshold tuning
            # =========================
            print(f"{'━' * 15} Optimize Probability Thresholds {'━' * 15}")
            t0 = time.time()
            thr_tensor = torch.as_tensor(prob_thresholds, device=device,
                                         dtype=torch.float32).view(1, -1)
            if thresholds_by_disease:

                all_predictions, all_probs, all_labels, all_patient_ids, val_loss_avg \
                    = validate_one_epoch(my_model, tune_loader, device, criterion, thr_tensor,
                                         use_amp)

                prob_thresholds = find_best_thresholds_per_label(probs=all_probs, labels=all_labels,
                                                                 n_grid=40)

                for label, thr in zip(top_labels, prob_thresholds):
                    print(f"{label}: {thr:.3f}")

        # Evaluation
        # =========================
        print(f"{'━' * 15} Evaluation {'━' * 15}")
        t0 = time.time()

        eval_results_training = evaluate_and_report(
            model=my_model,
            loader=tune_loader,
            device=device,
            criterion=criterion,
            prob_thresholds=prob_thresholds,
            use_amp=use_amp,
            derive_negatives=derive_negatives,
            top_labels=top_labels,
            NUM_EPOCHS=NUM_EPOCHS,
            epoch=epoch,
            train_loss_avg=train_loss_avg,
            give_stats=True,
            final_evaluation=False,
            title="Statistics on Tune Set"
        )

        val_loss_avg = eval_results_training["val_loss_avg"]
        fn_rate = eval_results_training["fn_rate"]
        f1_neg = eval_results_training["f1_neg"]
        f1_pos = eval_results_training["f1_pos"]

        dt = time.time() - t0
        print(f"Test Validation sec:", round(dt, 2))

        if epoch >= 1 and update_pos_weights:
            pos_weight = abs(1.0 / (1.0 - 0.2 * fn_rate + 1e-6))
            pos_weight = pos_weight.to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # if max(fp_rate) < 0.11 and (max(fn_rate) > 0.5 or np.mean(fn_rate) > 0.3):
        #     pos_weight = 1.0 / (1.0 - 0.1 * fn_rate + 1e-6)
        #     pos_weight = pos_weight.to(device)
        #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # else:
        #     criterion = nn.BCEWithLogitsLoss()

        if epoch < NUM_EPOCHS - 1:
            if f1_neg >= f1_neg_threshold and f1_pos > f1_pos_threshold:
                early_stopping = True
                print(f"Early stop at epoch {epoch + 1} because we reached "
                      f"threshold f1 neg {f1_neg} over {f1_neg_threshold} "
                      f"and f1 pos {f1_pos} over {f1_pos_threshold}.")

            if val_loss_avg > val_loss_min + delta_loss:
                early_stopping = True
                print(
                    f"Early stop at epoch {epoch + 1} because this epochs loss {val_loss_avg} "
                    f"compared to the minimal loss in the previous epochs {val_loss_min}.")
            else:
                val_loss_min = min(val_loss_avg, val_loss_min)

            # print(f"{'━' * 15} Statistics on Tune Set {'━' * 15}")
            # val_accuracy, eval_stats = give_epoch_stats(f1_neg, f1_pos, NUM_EPOCHS, epoch,
            #                                             top_labels, all_predictions, all_labels,
            #                                             train_loss_avg, val_loss_avg, all_probs,
            #                                             final_evaluation=False)

            if early_stopping:
                break

    if early_stopping or epoch == NUM_EPOCHS - 1:

        model_dir = os.path.join(".", "Models")
        os.makedirs(model_dir, exist_ok=True)
        save_model(my_model, optimizer, epoch, prob_thresholds, top_labels,
                   "./Models/model_standard.pt")

        # Placeholder fine-tuned model
        my_model_wft = None
        thr_tensor_wft = None
        # Finetuning
        # Fine tune the last layer only for those train_labels,
        # where fn / (fn + tp) = fn_rate >= 0.3,
        # bad_labels should just choose the id of the disease with too high fn_rates
        neg_idx = list(top_labels).index("Negative")
        bad_labels = [i for i in np.where(fn_rate > 0.4)[0].tolist() if i != neg_idx]
        if bad_labels:

            print(f"{'━' * 15} Fine Tuning {'━' * 15}")
            my_model_wft = copy.deepcopy(my_model)
            my_model_wft, optimizer_wft = fine_tune_bad_labels(my_model_wft, train_loader.dataset,
                                                               bad_labels, device,
                                                               partial_unfreeze_bad_labels,
                                                               pin_memory, n_epochs=2,
                                                               lr=1e-4)

            # Recalculate probability thresholds
            all_predictions, all_probs, all_labels, all_patient_ids, val_loss_avg \
                = validate_one_epoch(my_model_wft, tune_loader, device, criterion, thr_tensor,
                                     use_amp)

            prob_thresholds_wft = find_best_thresholds_per_label(probs=all_probs,
                                                                 labels=all_labels,
                                                                 n_grid=40)
            thr_tensor_wft = torch.as_tensor(prob_thresholds_wft,
                                             device=device, dtype=torch.float32).view(1, -1)

            for label, thr in zip(top_labels, prob_thresholds_wft):
                print(f"{label}: {thr:.3f}")

            save_model(my_model_wft, optimizer_wft, epoch, prob_thresholds_wft, top_labels,
                       "./Models/model_finetuned.pt")

        # Statistics
        # =========================
        print(f"{'━' * 15} Final Performance Evaluation {'━' * 15}")
        standard_results = evaluate_and_report(
            model=my_model,
            loader=test_loader,
            device=device,
            criterion=criterion,
            prob_thresholds=prob_thresholds,
            use_amp=use_amp,
            derive_negatives=derive_negatives,
            top_labels=top_labels,
            NUM_EPOCHS=NUM_EPOCHS,
            epoch=epoch,
            train_loss_avg=train_loss_avg,
            give_stats=True,
            final_evaluation=True,
            title="Statistics on Test Set for standard model"
        )

        if my_model_wft is not None:
            results_wft = evaluate_and_report(
                model=my_model_wft,
                loader=test_loader,
                device=device,
                criterion=criterion,
                prob_thresholds=prob_thresholds_wft,
                use_amp=use_amp,
                derive_negatives=derive_negatives,
                top_labels=top_labels,
                NUM_EPOCHS=NUM_EPOCHS,
                epoch=epoch,
                train_loss_avg=train_loss_avg,
                give_stats=True,
                final_evaluation=True,
                title="Statistics on Test Set for fine tuned model"
            )

    end = time.perf_counter()

    print(f"Elapsed for train/eval: {end - start:.4f} seconds")
    #############################################
    # 6) Save Statistics and Plot
    #############################################
    print('Saving statistics')

    start1 = time.perf_counter()
    eval_dir = os.path.join(".", "Evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    save_and_plot(standard_results, eval_dir, top_labels, my_model, test_loader, device,
                  prob_thresholds, mean, std, derive_negatives, name='_std')
    if my_model_wft is not None:
        save_and_plot(results_wft, eval_dir, top_labels, my_model_wft, test_loader, device,
                      prob_thresholds_wft, mean, std, derive_negatives, name='_wft')
    end1 = time.perf_counter()
    print(f"Elapsed for stats/pics: {end1 - start1:.1f} seconds")
