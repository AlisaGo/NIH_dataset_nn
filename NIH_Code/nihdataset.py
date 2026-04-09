"""
nihdataset.py

Custom Dataset and preprocessing utilities for NIH Chest X-Ray images:

Classes:
  NIHChestXRayDataset:
    __init__:
      - Load NIH metadata CSV and image file paths.
      - When use_cache is set to true, load jpeg images from cache directory.
      - One-hot encode each finding label into its own column.
      - Optionally select only the top `top_x` labels.
      - Build either:
          • a full train / tune / test split
          • or representative train / tune / test subsets when `max_size` is smaller than the dataset.
      - Build an internal list of (image_path, label_vector, patient_id, split) tuples,
        where split is one of "train", "tune", or "test".

    __len__:
      - Return the number of samples in the dataset.

    __getitem__:
      - Given an index, load the corresponding image, apply split-specific transforms,
        and return image tensor + label tensor + patient ID.

    give_nih_dataset:
      - Read `Data_Entry_2017.csv`, map each “Image Index” to its filesystem path,
        and store only valid entries with an existing image file.

    set_binary_labels:
      - Rename “No Finding” to “Negative” and split the multi-label string into individual binary columns.
      - Compute and store the overall label frequency distribution.
      - Print image-level and patient-level label statistics.

    remove_noisy_images:
      - Filter out images whose Laplacian variance falls below a threshold (blurry or unreadable).

    is_noisy:
      - Compute the variance of the Laplacian on a grayscale image to detect blur;
        return True if the image is considered noisy or cannot be read.

    make_train_tune_val_split:
      - Build patient-wise train / tune / test splits.
      - Use either a random split or the official NIH split files when requested.

    give_official_split:
      - Read the official NIH train / validation and test file lists,
        and return the corresponding row indices.

    select_top_labels:
      - Keep only the top `top_x` most frequent labels and remove rows
        that do not contain any of those labels.

    build_representative_subset:
      - Construct a patient-wise representative subset of approximately `max_size`
        using the selected top labels and adjusted label proportions.

    sample_patients:
      - Sample patients without replacement until the requested number of images is reached,
        then return the selected rows and the remaining DataFrame.

    plot_disease_distribution:
      - Plot a bar chart of the most frequent “Finding Labels” strings for quick EDA.

  XRayStandardize:
    __init__:
      - Configure grayscale conversion, percentile clipping, and optional resizing.

    __call__:
      - Convert an X-ray to grayscale, apply percentile clipping and normalization,
        optionally resize it, and replicate it to RGB format.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from itertools import chain
import cv2
from PIL import Image


class NIHChestXRayDataset(Dataset):
    def __init__(self,
                 main_dir,
                 label_file,
                 use_official_split,
                 train_val_list_path=None,
                 test_list_path=None,
                 max_size=5000,
                 eval_frac=0.15,
                 top_x=6,
                 train_transform=None,
                 eval_transform=None,
                 excluded_labels=None,
                 cache_dir=None,
                 random_seed=42,
                 use_cache=False):

        self.random_seed = random_seed
        self.main_dir = main_dir
        self.label_file = label_file
        self.use_official_split = use_official_split
        self.train_val_list_path = train_val_list_path
        self.test_list_path = test_list_path
        self.top_x = top_x
        self.eval_frac = eval_frac
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.excluded_labels = set(excluded_labels or [])
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.original_df = []
        self.data_df = []
        self.all_labels = []
        self.top_labels = []
        self.label_distribution = []
        self.train_idx = []
        self.test_idx = []
        self.tune_idx = []

        # 1) Load NIH metadata (CSV) + image paths.
        self.give_nih_dataset()
        if max_size is None or max_size > len(self.data_df):
            max_size = len(self.data_df)

        self.max_size = max_size

        # 2) Encode train_labels into separate columns.
        self.set_binary_labels()

        # 3a)* Remove noisy images
        # self.remove_noisy_images()

        # # 3b) Subsample a representative subset
        # if max_size < len(self.data_df) or top_x < len(self.all_labels):
        #     self.build_representative_subset()
        # 3b) Select top train_labels
        self.select_top_labels()

        # 4) Subsample a representative subset
        if self.max_size is not None and self.max_size < len(self.data_df):
            if use_official_split:
                train_idx, test_idx = self.give_official_split()
                train_df = self.data_df.iloc[train_idx]
                test_df = self.data_df.iloc[test_idx]
            else:
                train_df = self.data_df
            eval_target = int(round(self.max_size * self.eval_frac))

            train_pool_df, _ = self.build_representative_subset(
                xray_df=train_df,
                max_size=self.max_size,
            )

            train_df, tune_df = self.split_train_tune_subset(
                train_pool_df,
                tune_frac=self.eval_frac,
                seed=self.random_seed,
                min_pos_per_label=5,
            )

            print("Train subset label sums:")
            print(train_df[list(self.top_labels)].sum())

            print("Tune subset label sums:")
            print(tune_df[list(self.top_labels)].sum())

            if use_official_split:
                test_df, _ = self.build_representative_subset(xray_df=test_df,
                                                              max_size=eval_target)

                self.data_df = pd.concat(
                    [
                        train_df.assign(split="train"),
                        tune_df.assign(split="tune"),
                        test_df.assign(split="test"),
                    ],
                    ignore_index=True
                )
                self.train_idx = self.data_df.index[self.data_df["split"] == "train"].to_numpy()
                self.tune_idx = self.data_df.index[self.data_df["split"] == "tune"].to_numpy()
                self.test_idx = self.data_df.index[self.data_df["split"] == "test"].to_numpy()
            else:
                self.data_df = train_pool_df.copy()
                self.make_train_tune_val_split()
        else:
            self.make_train_tune_val_split()

        # 5) Store the samples as (img_path, label_vector,patient_id) triples in a Python list
        sample_cols = ["path", "Patient ID", "split"] + list(self.top_labels)
        df_samples = self.data_df[sample_cols].copy()

        self.samples = [
            (
                row[0],  # path
                np.asarray(row[3:], dtype=np.float32),  # labels
                int(row[1]),  # Patient ID
                row[2],  # split
            )
            for row in df_samples.itertuples(index=False, name=None)
        ]

    def split_train_tune_subset(self, xray_df, tune_frac=0.15, seed=42, min_pos_per_label=10):
        """
        Split an already constructed representative train subset into
        train and tune parts patient-wise.

        Goals:
        - no patient leakage
        - approximate tune fraction
        - ensure at least a minimum number of positive samples per label in tune
        """
        top_labels = list(self.top_labels)
        rng = np.random.default_rng(seed)

        # patient-level label table
        patient_df = xray_df.groupby("Patient ID")[top_labels].max()

        all_patient_ids = patient_df.index.to_numpy()
        rng.shuffle(all_patient_ids)

        tune_patients = set()

        # ---- Stage 1: guarantee minimum positives per label in tune ----
        for label in top_labels:
            if label == "Negative":
                continue

            current_tune_df = xray_df[xray_df["Patient ID"].isin(tune_patients)]
            current_pos = int(current_tune_df[label].sum()) if len(current_tune_df) > 0 else 0

            max_possible = int(xray_df[label].sum())
            target_pos = min(min_pos_per_label, max_possible)

            if current_pos >= target_pos:
                continue

            candidate_pids = patient_df.index[
                (patient_df[label] == 1) & (~patient_df.index.isin(tune_patients))
                ].to_numpy()

            rng.shuffle(candidate_pids)

            for pid in candidate_pids:
                tune_patients.add(pid)
                current_tune_df = xray_df[xray_df["Patient ID"].isin(tune_patients)]
                current_pos = int(current_tune_df[label].sum())
                if current_pos >= target_pos:
                    break

        # ---- Stage 2: fill up to target tune size ----
        n_target = int(round(len(patient_df) * tune_frac))
        n_target = max(1, n_target)
        n_target = max(n_target, len(tune_patients))

        # rarity score: patients carrying rarer labels are prioritized
        label_counts = patient_df.sum(axis=0).astype(float)
        inv_freq = 1.0 / np.maximum(label_counts, 1.0)
        rarity_score = patient_df.mul(inv_freq, axis=1).sum(axis=1)

        remaining_pids = [pid for pid in all_patient_ids if pid not in tune_patients]
        remaining_pids = sorted(
            remaining_pids,
            key=lambda pid: rarity_score.loc[pid],
            reverse=True
        )

        for pid in remaining_pids:
            if len(tune_patients) >= n_target:
                break
            tune_patients.add(pid)

        tune_df = xray_df[xray_df["Patient ID"].isin(tune_patients)].copy()
        train_df = xray_df[~xray_df["Patient ID"].isin(tune_patients)].copy()

        return train_df, tune_df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetch a single sample from self.samples, using a single integer index.
        """
        img_path, label_vec, patient_id, split = self.samples[idx]

        # Load the image
        with Image.open(img_path) as im:
            image = im.convert("RGB")
        if split == "train" and self.train_transform is not None:
            image = self.train_transform(image)
        elif split in ("tune", "test") and self.eval_transform is not None:
            image = self.eval_transform(image)
        # Convert the label array/list to a tensor
        label_tensor = torch.from_numpy(label_vec)
        return image, label_tensor, patient_id

    def give_nih_dataset(self):
        """
        Load the official NIH CSV and match each row to an actual image path.
        """
        main_dir = self.main_dir
        all_xray_df = pd.read_csv(self.label_file)
        if self.use_cache == False:
            patterns = [
                os.path.join(main_dir, 'Data', 'archive', 'images_*/images', '*.png'),
            ]
        elif self.use_cache == True:
            patterns = [
                os.path.join(self.cache_dir, "*.jpg"),
                # os.path.join(main_dir, 'images_002/images', '*.png'),
            ]
        # Collect images from each pattern into one dictionary
        all_image_paths = {
            os.path.basename(x): x
            for pattern in patterns
            for x in glob(pattern)
        }

        print('Scans found:', len(all_image_paths), ', Total Headers:', all_xray_df.shape[0])
        # Map each Image Index to its full file path
        if self.use_cache:
            all_xray_df["cache_name"] = all_xray_df["Image Index"].str.replace(".png", ".jpg",
                                                                               regex=False)
            all_xray_df["path"] = all_xray_df["cache_name"].map(all_image_paths.get)
        else:
            all_xray_df["path"] = all_xray_df["Image Index"].map(all_image_paths.get)

        valid_xray_df = all_xray_df[all_xray_df['path'].notna()]

        # Number of patients
        num_patients = valid_xray_df["Patient ID"].nunique()
        print(f"Number of patients: {num_patients}")

        # Mean images per patient
        mean_images_per_patient = valid_xray_df.groupby("Patient ID").size().mean()
        print(f"Mean images per patient: {mean_images_per_patient:.2f}")

        self.original_df = valid_xray_df  # This stays
        self.data_df = valid_xray_df  # This is modified depending on max_size, different train
        # and validation splittings

    def set_binary_labels(self):
        """
        Replace 'No Finding' with 'Negative'.
        Encode each label into its own column with binary entries.
        Compute self.label_distribution as a sorted Series.
        """
        all_xray_df = self.data_df
        # Rename 'No Finding' -> 'Negative'
        all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(
            lambda x: x.replace('No Finding', 'Negative'))

        # Collect all unique train_labels by splitting on '|'
        all_labels = np.unique(
            list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist()))
        )

        # Filter out empty/meaningless strings and remove excluded train_labels if present
        all_labels = [lbl for lbl in all_labels if lbl not in self.excluded_labels]
        self.all_labels = all_labels
        # print('All Labels ({}): {}'.format(len(all_labels), all_labels))

        # Create a one-hot column for each label
        for c_label in all_labels:
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(
                lambda finding: 1 if c_label in finding.split('|') else 0
            )

        # If any disease label is positive, set Negative to 0.
        disease_cols = [c for c in all_labels if c != 'Negative']
        pos_rows = all_xray_df[disease_cols].sum(axis=1) > 0
        all_xray_df.loc[pos_rows, 'Negative'] = 0

        # Build a distribution (column sum) for each label, then sort descending
        label_distribution = {
            label: all_xray_df[label].sum() for label in all_labels
        }
        label_distribution = pd.Series(label_distribution).sort_values(ascending=False)
        self.label_distribution = label_distribution

        # Disease distribution, image level
        print("Diseases by frequency")
        print(label_distribution)

        # For each patient and label, detect whether the label changes across images
        patient_group = all_xray_df.groupby("Patient ID")[all_labels]
        label_changed = (patient_group.max() != patient_group.min()).astype(int)
        n_changed_labels_per_patient = label_changed.sum(axis=1)
        print("Patients by number of labels that change across their images")
        print(n_changed_labels_per_patient.value_counts().sort_index())

        # Disease distribution per patient
        # patient_level = patient_group.max()
        # label_distribution_patient = patient_level.sum().sort_values(ascending=False)
        # print("Diseases by frequency (patient level)")
        # print(label_distribution_patient)

    def remove_noisy_images(self, threshold=10):
        """
        Filter out 'noisy' or blurry images by checking the variance of the Laplacian.
        Increase 'threshold' to be more or less strict.
        """
        df = self.data_df
        df['Is_Noisy'] = df['path'].apply(lambda x: self.is_noisy(x, threshold=threshold))
        print("Before noise removal:", df.shape)
        print("After noise removal:", df[~df['Is_Noisy']].shape)
        self.data_df = df[~df['Is_Noisy']]

    def is_noisy(self, image_path, threshold=10):
        """
        Determine if an image is noisy based on variance of the Laplacian (focus measure).
        Returns True if the image is considered noisy (blurry), or if it couldn't be read.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return True  # If image path is invalid, treat as noisy
            variance = cv2.Laplacian(img, cv2.CV_64F).var()
            return variance < threshold
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return True

    def make_train_tune_val_split(self):
        nih_dataset = self.data_df
        use_official_split = self.use_official_split
        train_val_list_path = self.train_val_list_path
        test_list_path = self.test_list_path
        eval_frac = self.eval_frac
        tune_frac = eval_frac
        df = self.data_df.reset_index(drop=True)
        # 1) Get patient IDs
        patient_ids = df["Patient ID"].to_numpy()
        unique_patients = np.unique(patient_ids)

        if use_official_split == False:

            # 2) Shuffle patients
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(unique_patients)

            # 3) Split patients
            n_test = int(len(unique_patients) * eval_frac)
            test_patients = set(unique_patients[:n_test])

            n_tune = int(len(unique_patients) * tune_frac)
            tune_patients = set(unique_patients[n_test:n_test + n_tune])

            train_patients = set(unique_patients[n_test + n_tune:])

            # 4) Assign image indices based on patient membership
            test_idx = np.where(np.isin(patient_ids, list(test_patients)))[0]
            tune_idx = np.where(np.isin(patient_ids, list(tune_patients)))[0]
            train_idx = np.where(np.isin(patient_ids, list(train_patients)))[0]
        else:
            train_idx, test_idx = self.give_official_split()

            patient_ids_train = df.iloc[train_idx]["Patient ID"].unique()
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(patient_ids_train)
            n_tune = int(len(patient_ids_train) * tune_frac)
            patient_ids_tune = patient_ids_train[:n_tune]
            patient_ids_train = patient_ids_train[n_tune:]

            tune_idx = df.iloc[train_idx][
                df.iloc[train_idx]["Patient ID"].isin(patient_ids_tune)].index.to_numpy()
            train_idx = df.iloc[train_idx][
                df.iloc[train_idx]["Patient ID"].isin(patient_ids_train)].index.to_numpy()

        self.data_df["split"] = None
        self.data_df.loc[train_idx, "split"] = "train"
        self.data_df.loc[tune_idx, "split"] = "tune"
        self.data_df.loc[test_idx, "split"] = "test"

        self.train_idx = np.asarray(train_idx)
        self.tune_idx = np.asarray(tune_idx)
        self.test_idx = np.asarray(test_idx)

        print(f"Train patients: {len(set(patient_ids[train_idx]))}")
        print(f"Tune patients:   {len(set(patient_ids[tune_idx]))}")
        print(f"Eval patients:   {len(set(patient_ids[test_idx]))}")

    def give_official_split(self):

        train_val_list_path = self.train_val_list_path
        test_list_path = self.test_list_path
        df = self.data_df.reset_index(drop=True)
        if train_val_list_path is None or test_list_path is None:
            raise ValueError("Official split requested but split file paths are missing.")

        with open(train_val_list_path, "r") as f:
            train_val_names = {line.strip() for line in f if line.strip()}

        with open(test_list_path, "r") as f:
            test_names = {line.strip() for line in f if line.strip()}

        train_val_mask = df["Image Index"].isin(train_val_names)
        test_mask = df["Image Index"].isin(test_names)

        train_idx = df.index[train_val_mask].to_numpy()
        test_idx = df.index[test_mask].to_numpy()

        return train_idx, test_idx

    def select_top_labels(self):
        """
        1) Determine the top_x train_labels from self.label_distribution.
        """
        top_x = self.top_x

        # 1) The label_distribution is already a sorted Series from set_binary_labels().
        label_counts_sorted = self.label_distribution

        if top_x >= len(label_counts_sorted):
            top_x = len(label_counts_sorted)
            print(f'We will use all {top_x} train_labels')
            self.top_x = top_x
        else:
            print(f'We will use the top {top_x} train_labels.')

        # Take top top_x train_labels including negative
        top_labels = label_counts_sorted.index[0:top_x]
        self.top_labels = top_labels
        xray_df = self.data_df
        mask = xray_df[top_labels].any(axis=1)
        filtered_df = xray_df[mask]
        self.data_df = filtered_df

    def build_representative_subset(self, xray_df, max_size):
        """
        2) Based on max_size of the representative set, compute how many samples we want for each
        label (scaled_counts).
        3) Filter the dataset to only rows that have at least one of these top train_labels.
        4) Perform patient-wise weighted random sampling for each label and concatenate the subsets.

        Important note: We start with the least common pathologies and sample patient-wise.
        When we collect representatives for pathology i, since we deal with multi-label
        representatives, other pathologies with index i +- k for some k might be present.
        At each step we subtract all sampled pathologies from the scaled_count.
        However, as we sample patient wise, the representative subset might be slightly
        bigger than max_size.
        """
        if max_size is None:
            max_size = self.max_size

        top_x = self.top_x
        top_labels = self.top_labels
        label_counts_sorted = xray_df[top_labels].sum(axis=0).astype(int)

        # If there are twice as many negatives as the most common illness, set the fraction of
        # negatives to be twice as big as the fraction of the most common illness
        neg_idx = list(top_labels).index("Negative")
        assert (neg_idx == 0)
        relation_neg_pos = label_counts_sorted.iloc[0] / label_counts_sorted.iloc[1]
        if relation_neg_pos > 2:
            relation_neg_pos_used = 2
        else:
            relation_neg_pos_used = relation_neg_pos

        # 2) Compute scaled counts.
        #    The total count of the top train_labels in the dataset:
        sum_negatives_used = relation_neg_pos_used * label_counts_sorted.iloc[1]
        total_count_top = sum(label_counts_sorted.iloc[1:top_x]) + sum_negatives_used

        #    We'll skip the first label for fraction if we assume it is "Negative"
        label_counts_fraction = np.zeros(top_x, dtype=float)
        label_counts_fraction[0] = sum_negatives_used / total_count_top
        for i in range(1, top_x):
            label_counts_fraction[i] = label_counts_sorted.iloc[i] / total_count_top

        # Prepare an array to store how many samples each of the top train_labels should get
        scaled_counts = np.ceil(label_counts_fraction * max_size).astype(int)

        print(
            "We will construct a representative subset of approximately",
            np.sum(scaled_counts),
            "samples, trying to keep proportions as:\n",
            scaled_counts,
            "\nThe proportions may vary because sampling is done patient-wise."
        )

        # 3) Filter the DataFrame to only those rows that have at least one of the top train_labels.
        mask = xray_df[top_labels].any(axis=1)
        filtered_df = xray_df[mask]

        # 4) Sample for each of the top train_labels individually and concatenate.
        remaining_df = filtered_df.copy()
        frames = []

        sample_order = list(label_counts_sorted.sort_values(ascending=True).index)

        for label in sample_order:
            i = list(top_labels).index(label)
            subset_df = remaining_df[remaining_df[label] == 1]
            n_samples = min(scaled_counts[i], len(subset_df))

            if n_samples > 0:
                chosen_df, remaining_df, picked_images_len = self.sample_patients(
                    n_samples, subset_df, remaining_df, self.random_seed
                )

                if chosen_df is not None and len(chosen_df) > 0:
                    picked_counts = chosen_df[top_labels].sum(axis=0).to_numpy()
                    scaled_counts = np.maximum(0, scaled_counts - picked_counts)
                    frames.append(chosen_df)

        if len(frames) == 0:
            print('No data could be found for the chosen top train_labels.')
            self.data_df = filtered_df.iloc[0:0].copy()
            return

        df_representing = pd.concat(frames, axis=0).drop_duplicates()

        need = max_size - len(df_representing)

        if need > 0 and len(remaining_df) > 0:
            pos_pool = remaining_df[remaining_df["Negative"] == 0]
            neg_pool = remaining_df.drop(index=pos_pool.index)

            # Fill with positives first
            if need > 0 and len(pos_pool) > 0:
                chosen_df, remaining_df, picked_images_len = self.sample_patients(
                    need, pos_pool, remaining_df, self.random_seed
                )

                if chosen_df is not None and len(chosen_df) > 0:
                    df_representing = pd.concat(
                        [df_representing, chosen_df],
                        axis=0
                    )
                    need -= picked_images_len

            # Recompute pools after remaining_df changed
            if need > 0 and len(remaining_df) > 0:
                pos_pool = remaining_df[remaining_df["Negative"] == 0]
                neg_pool = remaining_df.drop(index=pos_pool.index)

            # Fill rest with negatives
            if need > 0 and len(neg_pool) > 0:
                chosen_df, remaining_df, picked_images_len = self.sample_patients(
                    need, neg_pool, remaining_df, self.random_seed
                )

                if chosen_df is not None and len(chosen_df) > 0:
                    df_representing = pd.concat(
                        [df_representing, chosen_df],
                        axis=0
                    )
                    need -= picked_images_len

        df_representing = df_representing.drop_duplicates().reset_index(drop=True)
        return df_representing, remaining_df

    @staticmethod
    def sample_patients(n_samples, subset_df, remaining_df, random_seed):
        if n_samples <= 0 or len(subset_df) == 0:
            return None, remaining_df, 0

        candidate_pids = subset_df["Patient ID"].drop_duplicates()
        candidate_pids = candidate_pids.sample(frac=1.0, random_state=random_seed).to_numpy()

        picked_pids = []
        picked_images = 0

        for pid in candidate_pids:
            pid_rows = remaining_df[remaining_df["Patient ID"] == pid]
            n_pid = len(pid_rows)

            picked_pids.append(pid)
            picked_images += n_pid

            if picked_images >= n_samples:
                break

        chosen_df = remaining_df[remaining_df["Patient ID"].isin(picked_pids)]
        remaining_df = remaining_df.drop(index=chosen_df.index, errors="ignore")

        return chosen_df, remaining_df, picked_images

    def plot_disease_distribution(self):
        """
        Example: quick bar plot of the top 15 multi-label strings from 'Finding Labels'.
        """
        all_xray_df = self.data_df
        label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
        ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
        ax1.set_xticklabels(label_counts.index, rotation=90)
        plt.show()


class XRayStandardize:
    def __init__(self, do_rescale=False, size=224, clip_percentiles=(1, 99)):
        self.do_rescale = do_rescale
        self.size = size
        self.clip_percentiles = clip_percentiles

    def __call__(self, image):
        # 1) grayscale
        arr = np.asarray(image.convert("L"), dtype=np.float32)

        # 2) percentile clipping
        lo, hi = np.percentile(arr, self.clip_percentiles)
        if hi > lo:
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo)
        else:
            arr = arr / 255.0

        # 3) back to PIL, resize, replicate to RGB
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        if self.do_rescale:
            img = img.resize((self.size, self.size), resample=Image.BICUBIC)
        img = Image.merge("RGB", (img, img, img))
        return img
