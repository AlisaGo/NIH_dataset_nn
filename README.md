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
# Performance

This project is remarkable, because we achieve competitive results using a lightweight setting:
- dataset caching reducing storage from ~45GB to ~1GB
- representative subset experiments on ~78k images in total
- split into train / tune / test using either:
  - official NIH split files (`USE_OFFICIAL_SPLIT=True`)
  - or patient-level random splitting (`USE_OFFICIAL_SPLIT=False`)
- in the benchmark runs, this resulted in an approximate 70% / 15% / 15% split

## Main benchmark setup

All experiments were run on the same general setup; 
only the parameters listed in the Results table were changed between runs.

Common setup:
- `SEED = 42`
- `use_cache = True`
- `USE_OFFICIAL_SPLIT = True`
- `EXCLUDED_LABELS = {"Nodule", "Mass", "Hernia"}`
- `MAX_IMAGES = 60000`
- `eval_frac = 0.15`
- `NUM_LABELS = 11`
- `THRESHOLD_TUNE_EPOCH = 1`
- `initial_prob_threshold = 0.25`
- `thresholds_by_disease = True`
- `derive_negatives = True`
- `NUM_EPOCHS = 3`
- `BATCH_SIZE = 32`
- `NUM_WORKERS = 2`
- `f1_neg_threshold = 0.85`
- `f1_pos_threshold = 0.4`
- `delta_loss = 0.02`
- `pretrained_model = "MultiLabelResNet"`
- `focal_gamma = 1.2`
- `want_save_model = False`

Parameters varied between runs:
- `lr`: `1e-4` ("fast learning") or `1e-5`
- `train_full_model`: `True` or `False`
- `freeze_mode_stage1`, `freeze_mode_stage2`, `unfreeze_epoch` when staged freezing was tested
- `bce_weight`: `0.95`, `0.9`, or `0.0` (pure focal loss)
- `pos_weight_type`
- `update_pos_weights`

In the experiments, no initial class weighting was used
(`pos_weight_type=None`), while dynamic positive weight updates were
optionally enabled via `update_pos_weights`.

## Results on test set

The following runs were compared on the same ~78k setup:

| Run config | Main change                                                          | Test F1 abs | Test F1 neg | Test F1 pos | Test AUROC |
|---|----------------------------------------------------------------------|---:|---:|---:|---:|
| lr=`1e-4`, train_full_model=True, bce_weight=0.95, pos_weight_type=None, update_pos_weights=True | BCE+Focal mix, no staged freezing, fast learning setup               | 32% | 50.94% | 39.60% | 78% |
| lr=`1e-4`, train_full_model=True, bce_weight=0.0, pos_weight_type=None, update_pos_weights=False | Pure Focal loss,fast learning                                        | 28% | 51.57% | 32.78% | 75% |
| lr=`1e-5`, train_full_model=False, bce_weight=0.9, unfreeze_epoch=2, pos_weight_type=None, update_pos_weights=False | static loss,Staged freezing (`head_only → last_block`),slow learning | 25% | 53.13% | 32.23% | 75% |
| lr=`1e-5`, train_full_model=True, bce_weight=0.9, pos_weight_type=None, update_pos_weights=False | static loss,slow learning                                            | 20% | 52.38% | 28.09% | 73% |

## Interpretation

- A BCE-dominant mixed loss (`bce_weight ≈ 0.95`) performs best in this short training regime.
- Switching to pure focal loss (`bce_weight = 0`) reduces overall performance, especially on positive classes.
- The staged freezing setup (`train_full_model=False`, unfreeze at epoch 2) underperforms in this 3-epoch setting, likely due to insufficient training time.
- The staged freezing result is not conclusive and may improve with:
  - more epochs  
  - lower learning rate (e.g. `1e-5` with longer training)
- The fine-tuning on bad labels didn't improve the AUC and F1 results and in some cases caused a 
  worsening of around 1%. This might be due to the fact that tuning on a subset of labels 
  damages the remaining features and should therefore be avoided.
- Even though the ROC curves show a clear signal, it remains hard to find a threshold which 
  gives a reliable binary decision. 
- Further experiments are left as follow-up due to limited time and compute resources.
Per-disease performance statistics, roc-curve plots, and sample classification are written to the `Evaluation/` directory.
------------------------------------------------------------------------

# Project Highlights

This project demonstrates a complete applied machine learning pipeline for medical imaging, with focus on robustness, efficiency, and realistic evaluation:

- Efficient dataset caching reducing storage from ~45GB to ~1GB, enabling fast local experimentation  
- Efficient training (few minutes per epoch on ~50k images on a single Colab GPU,
  depending on configuration)
- Patient-level dataset splitting to prevent data leakage across train, tune, and test sets
- Multi-label classification setup reflecting real clinical diagnosis scenarios  
- Transfer learning using pretrained CNN architectures (ResNet / MobileNet)  
- Handling of severe class imbalance through configurable sampling and loss weighting  
- Flexible loss design combining BCE and Focal Loss  
- Support for initializing positive class weights to emphasize rare pathologies  
- Dynamic update of positive weights based on the model's ability to separate
  positive and negative samples (gap-based weighting)
- Per-label threshold optimization to improve sensitivity-specific performance  
- Error-driven fine-tuning focused on underperforming disease labels  
- End-to-end pipeline from data preprocessing to evaluation with medical metrics  

------------------------------------------------------------------------
# Challenges

- Severe class imbalance across diseases
- Label noise (~90% estimated accuracy)
- Multi-label dependency between pathologies
- Trade-off between recall and precision in medical setting
- Although ROC curves show meaningful separation, translating this into stable
  binary decisions remains challenging, even with per-label threshold tuning.
- Strong limitations on computational power and time limited experiments

------------------------------------------------------------------------

# Repository Structure
```
NIH_dataset_nn/
├── Cached_Data/
│   └── cache_224_tot_90_jpg/
├── Evaluation/
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
├── requirements_frozen.txt
└── README.md
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

### Freezing behavior

Both models implement configurable freezing through `set_trainable(freeze_mode=...)`.

Supported options are:

- `freeze_mode="head_only"`  
  Freeze the full backbone and train only the classifier head.

- `freeze_mode="last_block"`  
  Train the classifier head and the last backbone block.

- `freeze_mode="last_two_blocks"`  
  Train the classifier head and the last two backbone blocks.

- `freeze_mode="full"`  
  Train the full model.

The exact mapping depends on the architecture:

- **ResNet**
  - `head_only` → only `fc`
  - `last_block` → `fc` + `layer4`
  - `last_two_blocks` → `fc` + `layer3` + `layer4`

- **MobileNet**
  - `head_only` → only `classifier`
  - `last_block` → `classifier` + `features[-1]`
  - `last_two_blocks` → `classifier` + `features[-2:]`

In all partial-freezing modes, the classifier head is always trainable.

--------------------------------------------------------------------------
# 5. Training Setup

Training is orchestrated in `analyze_nih.py` and uses helper functions
from `functions.py`.

### General setup flow

The training script performs the following steps:

1. choose device automatically (`cuda` / `mps` / `cpu`)  
2. build the selected model (`MultiLabelResNet` or `MultiLabelMobileNet`)  
3. configure the training mode (full training or staged freezing)  
4. prepare train and evaluation transforms  
5. construct `NIHChestXRayDataset` with the chosen split, labels, and subset size  
6. create `train`, `tune`, and `test` DataLoaders  
7. define loss function and optimizer  
8. train for `NUM_EPOCHS`  
9. optionally tune thresholds on the tune set  
10. optionally update positive loss weights during training  
11. optionally fine-tune difficult labels  
12. evaluate on the test set and save model, plots, and CSV outputs  

Core functions involved in training:

- `get_data_loaders()` -- builds train / tune / test DataLoaders from dataset indices
- `train_one_epoch()` -- performs one training epoch
- `validate_one_epoch()` -- runs model evaluation on tune or test data
- `find_best_thresholds_per_label()` -- tunes one threshold per label
- `fine_tune_bad_labels()` -- performs targeted fine-tuning on difficult labels

------------------------------------------------------------------------
## 5a. Training Configuration

The main experiment setup is defined in `analyze_nih.py`.

### Data and preprocessing

- `use_cache`  
  Uses cached JPEG images instead of original PNG files.  
  - `True` → faster training with preprocessed images  
  - `False` → loads original NIH images  

- `do_rescale`  
  Controls whether images are resized inside `XRayStandardize`.  
  This is automatically disabled when `use_cache=True`.

- `MAX_IMAGES`  
  Controls dataset size by building a representative subset.  
  - smaller → faster experiments  
  - `None` → use the full available dataset  

- `eval_frac`  
  Fraction used to build tune and evaluation subsets.

- `EXCLUDED_LABELS`  
  Removes specific diseases from the training task.

- `NUM_LABELS`  
  Number of active labels used after exclusion, selected from the most frequent labels.

### Split behavior

- `USE_OFFICIAL_SPLIT`  
  - `True` → uses the NIH official split files  
  - `False` → creates patient-level random splits internally  

### Threshold behavior

- `THRESHOLD_TUNE_EPOCH`  
  Epoch at which threshold tuning is performed.

- `initial_prob_threshold`  
  Initial threshold used before tuning.

- `thresholds_by_disease`  
  - `True` → optimize one threshold per label  
  - `False` → keep one shared threshold  

- `derive_negatives`  
  - `True` → apply consistency rules for the `Negative` label in relation to pathology 
    predictions
  - `False` → treat `Negative` like an independent label

### Training control

- `NUM_EPOCHS`  
  Number of main training epochs.

- `BATCH_SIZE`  
  Batch size used in all loaders.

- `NUM_WORKERS`  
  Number of DataLoader workers.  
  This may be increased automatically on CUDA.

- `f1_neg_threshold`  
  Minimum F1 target for the `Negative` class.

- `f1_pos_threshold`  
  Minimum weighted F1 target for positive disease labels.

- `delta_loss`  
  Allowed increase above the best validation loss before stopping.

### Model, freezing, and optimization

- `pretrained_model`  
  Selects the backbone:
  - `MultiLabelResNet`
  - `MultiLabelMobileNet`

- `train_full_model`
  - `True` → train all parameters from the beginning  
  - `False` → apply `freeze_mode_stage1` at the start of training

- `freeze_mode_stage1`, `freeze_mode_stage2`, `unfreeze_epoch`  
  Control staged unfreezing when partial freezing is used.

The optimizer and learning rate are defined in `analyze_nih.py` and depend on
the selected training mode and experiment. In the benchmark runs described in
this repository, two main regimes were tested:

- **fast learning**: `lr = 1e-4`
- **slow learning**: `lr = 1e-5`

For serious experiments, `MAX_IMAGES` should be at least around `30000-50000`
to guarantee enough samples for learning each pathology. It's important to note that 
`MAX_IMAGES` refers to the size of the train loader (70% of the total) and not to the total size of 
the images used 
for training, tuning and evaluation.
------------------------------------------------------------------------
# 6. Loss and Class Imbalance

Medical multi-label classification is strongly affected by class imbalance.
Most images contain no findings, while some diseases are rare.
This can cause the model to favor negative predictions and under-detect positive cases.

To address this, the training combines a mixed loss with optional positive
class weighting.

## Loss configuration

The training uses a combination of Binary Cross Entropy (BCE) and Focal Loss.

- `bce_weight` controls the mixture:
  - `1.0` → pure BCE
  - `0.0` → pure Focal loss
  - intermediate values → weighted combination

- `focal_gamma` controls how strongly hard examples are emphasized

In practice:

- BCE provides stable optimization
- Focal loss increases the contribution of harder examples
- the final loss is the weighted sum of BCE and Focal loss

In the benchmark experiments, a BCE-dominant mix performed better than pure
Focal loss.

## Positive class weighting

The loss also supports positive re-weighting through `pos_weight`, which affects
the BCE part of the loss.

### Initial weighting

Initial weights are controlled by `pos_weight_type`:

- `None` → no initial class weighting (`pos_weight = 1` for all labels)
- `"sqrt_ratio"` → uses a softer version of the imbalance ratio
- `"prefer_rarest"` → gives stronger emphasis to rarer labels

If enabled, the weights are computed from the train-label frequencies and then
clipped by `pos_weight_cap`.

### Dynamic weight update

If `update_pos_weights = True`, the positive weights are updated after each
validation step.

This update is **not based directly on false negative rate**. Instead, for each
label the loss computes:

- `mu_1`: the mean predicted probability on samples where the label is positive
- `mu_0`: the mean predicted probability on samples where the label is negative

From these, it defines a separation gap:

- `gap = mu_1 - mu_0`

This gap measures how well the model separates positive and negative samples for
that label:

- large gap → positives and negatives are already well separated
- small gap → the label is still hard to distinguish

The target positive weight is then increased when the gap is small:

- small gap → larger `pos_weight`
- large gap → `pos_weight` stays closer to `1`

The update is smoothed with momentum, so the weights do not jump abruptly from
one epoch to the next.

Note: The dynamic positive weighting mechanism operates independently from
the focal loss component and directly modifies the `pos_weight` used in
`BCEWithLogitsLoss`.

### Intuition

This mechanism does not only look at how rare a label is in the dataset.
Instead, it reacts to how well the model is currently separating positives from
negatives for each label.

This makes the weighting adaptive:

- labels that remain poorly separated receive stronger emphasis
- labels that are already learned well are not over-penalized

So the update is driven by the model's current confidence structure, not only by
static class frequencies.

## Practical observations

From the benchmark experiments:

- imbalance handling might have a small positive impact on positive-class performance and 
  hence on the positive weighting reduced F1. Due to computational and time restriction its 
  impact was not fully verified but seemed a priori rather small (~1-2%)
- pure Focal loss did not outperform the BCE-dominant mixed loss
- a mixed BCE/Focal setup with adaptive weighting gave the most stable results
  in short runs
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
4. searches thresholds on a fixed grid (e.g`0.25`-`0.7`)  
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

If `derive_negatives=True`, the `Negative` label is handled with a
competition-based rule instead of being treated like an ordinary independent
label.

The prediction logic works as follows:

1. all labels are first thresholded normally  
2. the `Negative` probability is compared against:
   - its own threshold
   - the maximum disease probability
   - the sum of the top disease probabilities  
3. `Negative` is predicted only if it is strong enough relative to the disease
   probabilities  
4. if no disease label exceeds threshold, `Negative` is set to `1` to avoid an
   all-zero prediction  
5. if `Negative` is predicted, all disease labels are forced to `0`

This means the `Negative` label is not treated as an independent class.
Instead, it competes with disease predictions and is also used as a fallback
when no disease is predicted.

If `derive_negatives=False`:

- all labels are thresholded directly
- if any disease label is predicted, `Negative` is forced to `0` to keep the
  output logically consistent

------------------------------------------------------------------------

# 9. Model Evaluation

Metrics are computed in `performance_metrics.py`.

Key metrics:

-   AUROC
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

After the main training stage, the project supports an optional fine-tuning step
focused on underperforming labels.

The goal is to re-train the model on samples containing labels that are currently
hard to detect, based on their false negative rates.

Function:

`fine_tune_bad_labels()`

## Configuration

Error-driven fine-tuning is configurable in both `analyze_nih.py` and `functions.py`.

### Main options

- `partial_unfreeze_bad_labels`  
  Controls whether fine-tuning updates:
  - only the classifier head  
  - or also part of the backbone  

- `n_epochs`  
  Number of fine-tuning epochs  

- `lr`  
  Learning rate used during fine-tuning  

---

## How fine-tuning works

After the main training loop:

1. the model is evaluated on the validation set  
2. the false negative rate (`fn_rate`) is computed for each label  
3. labels with high FN rate are selected (e.g. `fn_rate > 0.30`)  

Then:

4. all samples containing at least one of these labels are collected  
5. a reduced dataset and DataLoader are created from these samples  
6. the model is fine-tuned only on this subset  

This creates a targeted training phase where the model focuses on labels it
currently struggles to detect.

---

## Training behavior

During fine-tuning:

- a simplified loss is used (`BCEWithLogitsLoss`)  
- an Adam optimizer is applied with a small learning rate (e.g. `1e-5`)  
- training runs for a small number of epochs  

Unlike the main training loop, this phase does not use the full loss
configuration (e.g. no focal component or dynamic re-weighting).

---

## Backbone behavior

Inside `fine_tune_bad_labels()`:

- all parameters are frozen first  
- the classifier head is always unfrozen  

If `partial_unfreeze_bad_labels=True`, part of the backbone is also unfrozen:

- **ResNet** → `layer4`  
- **MobileNet** → last feature blocks (`features[-2:]`)  

---

## Practical observations

In the benchmark experiments:

- labels with `fn_rate > 0.30` were selected  
- `n_epochs = 1`  
- `lr = 1e-5`  
- `partial_unfreeze_bad_labels = False`  

This fine-tuning step did **not improve performance** and in some cases
slightly degraded it.

A likely reason is that focusing on a subset of labels can distort previously
learned feature representations and harm overall generalization.
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

The models can be saved using 
-   `save_model()`

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

A frozen dependency snapshot is also included in `requirements_frozen.txt`.

------------------------------------------------------------------------

# Data Layout

The code expects this structure:

Original Data
```
Data/archive/images_001/images
Data/archive/images_002/images
...
Data/archive/images_012/images
```

Labels and splits
```
Information/
├── Data_Entry_2017.csv
├── train_val_list.txt
└── test_list.txt
```
Cached Data
```
Cached_Data/ cache_224_tot_90_jpg/
```
------------------------------------------------------------------------

# Running the Project

Build the image cache:
```
python NIH_Code/build_cache.py
```

Run training and evaluation:
```
python NIH_Code/analyze_nih.py
```
Outputs will appear in the `Evaluation/` directory.

If chosen, a saved trained model will appear in the `Model/` directory.

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
  - tuning on loss function
-   dataset engineering
-   class imbalance handling
-   model evaluation and metrics
-   applied machine learning for healthcare


