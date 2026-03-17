# NIH Chest X-Ray Multi-Label Classification with Transfer Learning

A PyTorch project for **multi-label disease classification on the NIH
Chest X-Ray dataset** using transfer learning, patient-aware dataset
splitting, threshold tuning, and targeted fine‑tuning.

This repository demonstrates a full applied machine learning workflow:

-   dataset preprocessing and caching
-   custom multi‑label dataset creation
-   transfer learning with pretrained CNNs
-   class imbalance handling
-   per-label threshold optimization
-   evaluation with medical classification metrics

The goal is to classify chest X‑ray images into **multiple possible
diseases simultaneously**, reflecting how medical diagnoses work in real
clinical settings.

------------------------------------------------------------------------

# Project Highlights

This project demonstrates practical machine learning engineering skills:

• Significant storage size reduction of the image dataset using cached data without significant 
performance loss\
• Medical imaging classification using deep learning\
• Transfer learning with pretrained CNN architectures\
• Multi‑label classification design\
• Patient‑level dataset splitting to avoid data leakage\
• Class imbalance handling through configurable loss weighting and representative sampling\
• Threshold optimization for each disease label\
• Error‑driven model improvement through targeted fine‑tuning

------------------------------------------------------------------------

# Repository Structure
```
NIH_dataset_nn/
├── Cached_Data/
│   └── cache_224_tot_90_jpg/
├── Information/
│   ├── Data_Entry_2017.csv
│   ├── train_val_list.txt
│   └── test_list.txt
├── NIH_Code/
│   ├── analyze_nih.py
│   ├── build_cache.py
│   ├── functions.py
│   ├── models.py
│   ├── nihdataset.py
│   └── performance_metrics.py
├── requirements.txt
└── README.md
├── Evaluation/
```
------------------------------------------------------------------------

# Dataset structure

NIH Chest X-Ray Dataset:

```
Original resolution: 1024x1024
Scans: 112120
Number of patients: 30805
Mean images per patient: 3.64

Diseases by frequency:
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
```

The scans for each patient are taken at different times (follow-ups), which results in labels being changed for the same patient. However, some labels may be noisy, as the dataset is estimated to be approximately 90% accurate.

```
Patients by number of labels that change across their images:
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
```

# End‑to‑End Pipeline

Depending on the configuration, the training pipeline may:

- exclude specific labels
- use a reduced representative subset
- operate on fewer labels than the original 14

At runtime, the script prints:

- dataset size
- patient statistics
- label frequencies
- label variability across patients

## 1. Image Preprocessing

The script `build_cache.py` prepares the dataset by:

-   scanning the NIH dataset folders
-   converting PNG images to RGB
-   resizing images while preserving aspect ratio and padding to **224 × 224**
-   saving optimized JPEG versions

This reduces disk usage and significantly speeds up training runs.
This allows to test and deploy the machine learning framework on local computers,
which is very time-consuming if done on the original dataset.
The memory used is reduced from 45GB to slightly less than 1GB if size=224, quality=90 is used.

Main function:

`build_cache(src_root, cache_dir, size=224, quality=90, workers=6)`

------------------------------------------------------------------------

# 2. Dataset Construction

The class `NIHChestXRayDataset` in `nihdataset.py` performs:

- metadata loading from `Data_Entry_2017.csv`
- mapping image names to real file paths
- converting label strings to binary multi-label vectors
- renaming **"No Finding" → "Negative"**
- selecting the most relevant labels based on the current configuration
- building a representative dataset subset that preserves approximate pathology proportions
- creating patient-level train / tune / test splits
- supporting both official NIH split files and internally generated patient-level splits

Key methods:

- `give_nih_dataset()` -- loads metadata and maps image names to file paths
- `set_binary_labels()` -- converts label strings into binary columns
- `select_top_labels()` -- selects the labels used in the current experiment
- `build_representative_subset()` -- creates a representative subset preserving label proportions
- `sample_patients()` -- performs patient-level sampling without leakage
- `make_train_tune_val_split()` -- creates patient-level train / tune / test splits
- `give_official_split()` -- loads the official NIH train/test split from split files
- `__getitem__()` -- loads images and labels

------------------------------------------------------------------------

# 3. Patient-Level Data Splitting

The dataset class (`NIHChestXRayDataset`) prepares patient-aware splits
(train, tune, test) to avoid data leakage.

The current configuration uses the **official NIH split files**:

- `train_val_list.txt`
- `test_list.txt`

The function `get_data_loaders()` wraps these indices into PyTorch
DataLoaders for training, tuning, and evaluation.

------------------------------------------------------------------------

# 4. Model Architecture

Two transfer learning models are available:

### MultiLabelMobileNet

Uses an ImageNet-pretrained MobileNetV2 backbone.
The final linear layer of the classifier is replaced to output `num_labels` logits.

### MultiLabelResNet

Uses an ImageNet-pretrained ResNet50 backbone.
The final fully connected layer is replaced to output `num_labels` logits.

In both cases, `num_labels` is determined by the current training configuration,
so the output size matches the labels actually used in the experiment.

------------------------------------------------------------------------
# 5. Training Setup

Training is orchestrated in `analyze_nih.py` and uses helper functions
from `functions.py`.

### General setup flow

The training script performs the following steps:

1. choose device automatically (`cuda` / `mps` / `cpu`)  
2. build the selected model (`MultiLabelResNet` or `MultiLabelMobileNet`)  
3. prepare train and evaluation transforms  
4. construct `NIHChestXRayDataset` with the chosen split, labels, and subset size  
5. create `train`, `tune`, and `test` DataLoaders  
6. define loss function and optimizer  
7. train for `NUM_EPOCHS`  
8. optionally tune thresholds on the tune set  
9. optionally update loss weights during training  
10. optionally fine-tune difficult labels  
11. evaluate on the test set and save plots / CSV outputs  

Core functions involved in training:

- `get_data_loaders()` -- builds train / tune / test DataLoaders from dataset indices
- `train_one_epoch()` -- performs one training epoch
- `validate_one_epoch()` -- runs model evaluation on tune or test data
- `find_best_thresholds_per_label()` -- tunes one threshold per label
- `fine_tune_bad_labels()` -- performs targeted fine-tuning on difficult labels

The script supports depending on the selected configuration:

- full-model training or classifier-head-only training
------------------------------------------------------------------------

# 5a. Training Configuration Guide

The main experiment setup is defined in `analyze_nih.py`.

#### Data and preprocessing

- `use_cache`  
  Uses cached JPEG images instead of original PNG files.  
  - `True` → faster training, fixed image size from cache  
  - `False` → loads original NIH images  

- `do_rescale`  
  Controls whether images are resized inside `XRayStandardize`.  
  This is automatically disabled when `use_cache=True`.

- `MAX_IMAGES`  
  Controls dataset size by building a representative subset.  
  - smaller → faster experiments  
  - `None` → use full available dataset  

- `eval_frac`  
  Fraction used to build tune and evaluation subsets.

- `EXCLUDED_LABELS`  
  Removes specific diseases from the training task.

- `NUM_LABELS`  
  Number of active labels used after exclusion, selected from the most frequent labels.

#### Split behavior

- `USE_OFFICIAL_SPLIT`  
  - `True` → uses NIH official split files  
  - `False` → creates patient-level random splits internally  

#### Threshold behavior

- `THRESHOLD_TUNE_EPOCH`  
  Epoch at which threshold tuning is performed.

- `initial_prob_threshold`  
  Initial threshold used before tuning.

- `thresholds_by_disease`  
  - `True` → optimize one threshold per label  
  - `False` → keep the initial shared threshold  

- `derive_negatives`  
  - `True` → assign the `Negative` label when no disease exceeds threshold  
  - `False` → keep raw thresholded outputs only  

#### Epoch and loader settings

- `NUM_EPOCHS`  
  Number of main training epochs.

- `BATCH_SIZE`  
  Batch size used in all loaders.

- `NUM_WORKERS`  
  Number of DataLoader workers.  
  This is increased automatically on CUDA.

#### Early stopping / stopping criteria

- `f1_neg_threshold`  
  Minimum F1 target for the `Negative` class.

- `f1_pos_threshold`  
  Minimum weighted F1 target for positive disease labels.

- `delta_loss`  
  Allowed increase above the best validation loss before stopping.

#### Model and optimization

- `pretrained_model`  
  Selects the backbone:
  - `MultiLabelResNet`
  - `MultiLabelMobileNet`

- `train_full_model`  
  - `True` → train all model parameters  
  - `False` → train classifier head only  

When `train_full_model=True`:
- learning rate = `1e-4`
- weight decay = `1e-4`

When `train_full_model=False`:
- learning rate = `1e-3`
- weight decay = `0.0`

#### Reproducibility and hardware

- `SEED`  
  Global random seed used for reproducibility.

- automatic device selection  
  The script uses:
  - CUDA if available
  - otherwise MPS
  - otherwise CPU

- `use_amp`  
  Mixed precision is enabled automatically only on CUDA.

### Default setup in the current script

The current default configuration is:

- `use_cache = True`
- `USE_OFFICIAL_SPLIT = True`
- `MAX_IMAGES = 2000`
- `eval_frac = 0.15`
- `NUM_LABELS = 11`
- `EXCLUDED_LABELS = {"Nodule", "Mass", "Hernia"}`
- `NUM_EPOCHS = 3`
- `BATCH_SIZE = 32`
- `pretrained_model = "MultiLabelResNet"`
- `train_full_model = True`
- `thresholds_by_disease = True`
- `derive_negatives = True`
------------------------------------------------------------------------

# 6. Class Imbalance Handling

Medical datasets are highly imbalanced. The training pipeline supports
optional positive class weighting to reduce majority-class bias.
This prevents the model from simply predicting the majority negative
class.

## Configuration

Class imbalance handling is controlled in `analyze_nih.py`.

### Main options

- `preset_pos_weights`  
  - `True` → computes positive class weights once from the training labels before training  
  - `False` → uses standard `BCEWithLogitsLoss()` without class weights  

- `update_pos_weights`  
  - `True` → updates `pos_weight` during training based on false negative rates  
  - `False` → keeps the original loss weighting fixed  

### How preset weights are computed

If `preset_pos_weights=True`, the script computes:

- `pos = number of positives per label`
- `N = number of training samples`
- raw weight = `(N - pos) / pos`
- clipped and smoothed with `log(weight) + 1`

These weights are then passed to:

`BCEWithLogitsLoss(pos_weight=pos_weight)`

### Dynamic weight update during training

If `update_pos_weights=True`, after validation the script updates the loss using:

- the current false negative rate per label

This makes rare or under-detected labels contribute more strongly to the loss during later epochs.

### Current default behavior

In the current script:

- `preset_pos_weights = False`
- `update_pos_weights = False`

So the default setup uses:

`BCEWithLogitsLoss()`

without additional positive weighting.

------------------------------------------------------------------------

# 7. Threshold Optimization

The project supports configurable probability thresholds and can optimize
an individual threshold for each label instead of relying on a single
shared threshold.

Function:

`find_best_thresholds_per_label()`

## Configuration

Threshold tuning is configurable in `analyze_nih.py`.

### Main options

- `THRESHOLD_TUNE_EPOCH`  
  Defines at which epoch threshold optimization is run.

- `initial_prob_threshold`  
  Starting threshold used before any tuning.

- `prob_thresholds`  
  Array of thresholds initialized from `initial_prob_threshold`.

- `thresholds_by_disease`  
  - `True` → optimize a separate threshold for each label  
  - `False` → keep the shared initial threshold for all labels  

### How tuning works

If `thresholds_by_disease=True`, the script:

1. evaluates the model on the tune set  
2. collects predicted probabilities and ground-truth labels  
3. runs `find_best_thresholds_per_label()`  
4. searches thresholds on a grid from `0.1` to `0.6`  
5. chooses the threshold with the best F1 score for each label  

The tuned thresholds are then used for later validation and final test evaluation.

### Current default behavior

In the current script:

- `THRESHOLD_TUNE_EPOCH = 1`
- `initial_prob_threshold = 0.25`
- `thresholds_by_disease = True` \
So thresholds are first initialized to `0.25`, then optimized per label after epoch 1.
------------------------------------------------------------------------

# 8. Prediction Logic

Predictions are generated by comparing sigmoid probabilities against the
configured thresholds.

The prediction stage also supports a special handling of the `Negative`
label, so that it is assigned consistently with the absence of predicted
disease labels.

Function:

`give_predictions(probs, thresholds, labels)`

## Configuration

Prediction behavior is controlled by both threshold settings and negative-label handling.

### Main options

- `prob_thresholds`  
  Per-label thresholds used to convert probabilities into binary predictions.

- `derive_negatives`  
  Controls how the `Negative` label is assigned.

### Prediction flow

During validation and testing:

1. the model outputs raw logits  
2. logits are converted to probabilities using `sigmoid`  
3. probabilities are compared against `prob_thresholds`  
4. binary predictions are created  

### Negative label handling

If `derive_negatives=True`:

- the index of the `Negative` label is found
- all other disease labels are checked
- `Negative` is set to 1 only if no disease label is predicted as positive

This prevents contradictory outputs such as:

- `Negative = 1`
- and a disease label = 1

at the same time.

If `derive_negatives=False`:

- all labels, including `Negative`, are used only through direct thresholding

### Current default behavior

In the current script:

- `derive_negatives = True`

So the `Negative` label is derived from the absence of predicted disease labels, rather than treated as an independent thresholded disease output.

------------------------------------------------------------------------

# 9. Model Evaluation

Metrics are computed in `performance_metrics.py`.

Key metrics:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   False Negative Rate
-   False Positive Rate

Core functions:

-   `compute_tp_fp_tn_fn()`
-   `give_performance_metrics()`
-   `compute_fn_fp_rate()`
-   `compute_weighted_f1()`

------------------------------------------------------------------------

# 10. Error-Driven Fine-Tuning

After the main training stage, the project supports an additional
fine-tuning step focused on underperforming labels.

This step is designed to improve recall on difficult diseases by training
again on samples that contain labels with high false negative rates.

Function:

`fine_tune_bad_labels()`

## Configuration

Error-driven fine-tuning is configurable in both `analyze_nih.py` and `functions.py`.

### Main options

- `partial_unfreeze_bad_labels`  
  Controls whether fine-tuning updates:
  - only the classifier head
  - or also the last backbone block

- `n_epochs`  
  Number of fine-tuning epochs

- `lr`  
  Learning rate used during fine-tuning

### How fine-tuning is triggered

After the main training loop, the script computes false negative rates.

Labels are selected for fine-tuning if:

- `fn_rate > 0.30`

Then:

1. all samples containing at least one of these labels are selected  
2. a reduced DataLoader is built  
3. the model is fine-tuned on these labels only  

### Backbone behavior

Inside `fine_tune_bad_labels()`:

- all parameters are frozen first
- classifier head is always unfrozen
- if `partial_unfreeze=True`, the last backbone block is also unfrozen:
  - `layer4` for ResNet
  - `features[-2:]` for MobileNet

### Current default behavior

In the current script:

- labels with `fn_rate > 0.30` are selected
- `n_epochs = 2`
- `lr = 1e-4`
- `partial_unfreeze_bad_labels = True`

------------------------------------------------------------------------

# 11. Visualization and Output

After training the pipeline saves:

Evaluation directory:

-   ROC curve plots
-   classification examples
-   CSV file with disease‑level metrics

Functions:

-   `plot_roc_curve()`
-   `plot_images_classification()`

------------------------------------------------------------------------

# Installation

Clone git repository:
git clone git@github.com:AlisaGo/NIH_dataset_nn.git
cd NIH_dataset_nn

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Data Layout

The code expects this structure:

Original Data
Data/ archive/ images_001/images/ images_002/images/

Labels and splits
Information/
├── Data_Entry_2017.csv
├── train_val_list.txt
└── test_list.txt

Cached Data
Cached_Data/ cache_224_tot_90_jpg/

------------------------------------------------------------------------

# Running the Project

Build the image cache:

python NIH_Code/build_cache.py

Run training and evaluation:

python NIH_Code/analyze_nih.py

Outputs will appear in the `Evaluation/` directory.

Before running the project, review the configuration section at the top of
`NIH_Code/analyze_nih.py` to adjust dataset size, active labels, split mode,
training mode, and threshold behavior.

------------------------------------------------------------------------

# Potential Improvements

Future extensions could include:

-   full experiment configuration files
-   automated hyperparameter tuning
-   model checkpoint saving
-   changes in nn architecture
------------------------------------------------------------------------

# Key Skills Demonstrated

This project demonstrates:

-   PyTorch deep learning workflows
-   transfer learning
-   multi‑label classification
-   dataset engineering
-   class imbalance handling
-   model evaluation and metrics
-   applied machine learning for healthcare


