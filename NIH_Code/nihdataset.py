"""
nihdataset.py

Custom Dataset for NIH Chest X-Ray images:

Classes:
  NIHChestXRayDataset:
    __init__:
      - Load NIH metadata CSV and image file paths.
      - When use_cache is set to true, load jpeg images from cache directory.
      - One-hot encode each finding label into its own column.
      - Optionally remove noisy/blurry images.
      - Subsample a balanced, representative subset of size `max_size` across the top `top_x` labels.
      - Build an internal list of (image_path, label_vector) pairs.

    __len__:
      - Return the number of samples in the dataset.

    __getitem__:
      - Given an index, load the corresponding image, apply transforms, and return image tensor + label tensor.

    give_nih_dataset:
      - Read `Data_Entry_2017.csv`, map each “Image Index” to its filesystem path, and store valid entries.

    set_binary_labels:
      - Rename “No Finding” to “Negative” and split the multi-label string into individual binary columns.
      - Compute and store the overall label frequency distribution.

    remove_noisy_images:
      - Filter out images whose Laplacian variance falls below a threshold (blurry or unreadable).

    is_noisy:
      - Compute the variance of the Laplacian on a grayscale image to detect blur; returns True if noisy.

    build_representative_subset:
      - Identify the `top_x` most common labels and compute sampling fractions to balance them up to `max_size`.
      - Sample without replacement per label and concatenate into the final DataFrame.

    plot_disease_distribution:
      - Plot a bar chart of the top N “Finding Labels” frequencies for quick EDA.
"""


import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from itertools import chain
import cv2


class NIHChestXRayDataset(Dataset):
    def __init__(self, main_dir,
                 max_size=5000, top_x=6,
                 transform=None, cache_dir=None,
                 use_cache=False):

        self.main_dir = main_dir
        self.top_x = top_x
        self.transform = transform
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.data_df = []
        self.all_labels = []
        self.top_labels = []
        self.label_distribution = []

        # 1) Load NIH metadata (CSV) + image paths.
        self.give_nih_dataset()
        if max_size is None or max_size > len(self.data_df):
            max_size = len(self.data_df)

        self.max_size = max_size

        # 2) Encode labels into separate columns.
        self.set_binary_labels()

        # 3a)* Remove noisy images
        # self.remove_noisy_images()

        # 3b) Subsample a representative subset
        if max_size < len(self.data_df) or top_x < len(self.all_labels):
            self.build_representative_subset()

        # 4) Store final data in self.data_df
        self.data_df.reset_index(drop=True, inplace=True)

        # 5) Store the samples as (img_path, label_vector) pairs in a Python list
        self.samples = [
            (
                row["path"],
                row[self.top_labels].to_numpy(dtype=np.float32)
            )
            for _, row in self.data_df.iterrows()
        ]
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetch a single sample from self.samples, using a single integer index.
        """
        img_path, label_vec = self.samples[idx]

        # Load the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert the label array/list to a tensor
        label_tensor = torch.from_numpy(label_vec)
        return image, label_tensor

    def give_nih_dataset(self):
        """
        Load the official NIH CSV and match each row to an actual image path.
        """
        main_dir = self.main_dir
        all_xray_df = pd.read_csv(os.path.join(main_dir, "../Information/Data_Entry_2017.csv"))
        if self.use_cache == False:
            patterns = [
                os.path.join(main_dir, 'images_*/images', '*.png'),
                # os.path.join(main_dir, 'images_002/images', '*.png'),
            ]
        elif self.use_cache == True:
            patterns = [
                os.path.join(self.cache_dir,"*.jpg"),
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
            all_xray_df["cache_name"] = all_xray_df["Image Index"].str.replace(".png", ".jpg", regex=False)
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

        self.data_df = valid_xray_df

    def set_binary_labels(self):
        """
        Replace 'No Finding' with 'Negative'.
        Encode each label into its own column with binary entries.
        Compute self.label_distribution as a sorted Series.
        """
        all_xray_df = self.data_df
        # Rename 'No Finding' -> 'Negative'
        all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', 'Negative'))

        # Collect all unique labels by splitting on '|'
        all_labels = np.unique(
            list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist()))
        )

        all_labels = [lbl for lbl in all_labels if len(lbl) > 1]  # Filter out empty/meaningless strings
        self.all_labels = all_labels
        # print('All Labels ({}): {}'.format(len(all_labels), all_labels))

        # Create a one-hot column for each label
        for c_label in all_labels:
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(
                lambda finding: 1 if c_label in finding else 0
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

        # Disease distribution
        patient_level = all_xray_df.groupby("Patient ID")[all_labels].max()

        label_distribution_patient = patient_level.sum().sort_values(ascending=False)
        print("Diseases by frequency (patient level)")
        print(label_distribution_patient)

        self.label_distribution = label_distribution

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

    def build_representative_subset(self):
        """
        1) Determine the top_x labels from self.label_distribution.
        2) Based on max_size of the representative set, compute how many samples we want for each
        label (scaled_counts).
        3) Filter the dataset to only rows that have at least one of these top labels.
        4) Perform patient-wise weighted random sampling for each label and concatenate the subsets.

        Important note: We start with the least common pathologies and sample patient-wise.
        When we collect representatives for pathology i, since we deal with multi-label
        representatives, other pathologies with index i +- k for some k might be present.
        At each step we subtract all sampled pathologies from the scaled_count.
        However, as we sample patient wise, the representative subset might be slightly
        bigger than max_size.
        """

        all_xray_df = self.data_df
        max_size = self.max_size
        top_x = self.top_x

        # 1) The label_distribution is already a sorted Series from set_binary_labels().
        label_counts_sorted = self.label_distribution

        if top_x >= len(label_counts_sorted):
            top_x = len(label_counts_sorted)
            print(f'We will use all {top_x} labels')
            self.top_x = top_x
        else:
            print(f'We will use the top {top_x} labels.')

        # Take top top_x labels including negative
        top_labels = label_counts_sorted.index[0:top_x]
        self.top_labels = top_labels

        # If there are twice as many negatives as the most common illness, set the fraction of
        # negatives to be twice as big as the fraction of the most common illness
        neg_idx = list(top_labels).index("Negative")
        assert(neg_idx == 0)
        relation_neg_pos = label_counts_sorted.iloc[0] /label_counts_sorted.iloc[1]
        if relation_neg_pos > 2:
            relation_neg_pos_used = 2
        else:
            relation_neg_pos_used = relation_neg_pos

        # 2) Compute scaled counts.
        #    The total count of the top labels in the dataset:
        sum_negatives_used = relation_neg_pos_used * label_counts_sorted.iloc[1]
        total_count_top = sum(label_counts_sorted.iloc[1:top_x]) + sum_negatives_used

        #    We'll skip the first label for fraction if we assume it is "Negative"
        label_counts_fraction = np.zeros(top_x, dtype=float)
        label_counts_fraction[0] = sum_negatives_used / total_count_top
        for i in range(1, top_x):
            label_counts_fraction[i] = label_counts_sorted.iloc[i] / total_count_top

        # Prepare an array to store how many samples each of the top labels should get
        scaled_counts = np.ceil(label_counts_fraction * max_size).astype(int)

        print(
            "We will construct a representative subset of approximately",
            np.sum(scaled_counts),
            "samples, trying to keep proportions as:\n",
            scaled_counts,
            "\nThe proportions may vary because sampling is done patient-wise."
        )

        # 3) Filter the DataFrame to only those rows that have at least one of the top labels.
        mask = all_xray_df[top_labels].any(axis=1)
        filtered_df = all_xray_df[mask]

        # 4) Sample for each of the top labels individually and concatenate.
        remaining_df = filtered_df.copy()
        frames = []

        for i, label in reversed(list(enumerate(top_labels))):
            subset_df = remaining_df[remaining_df[label] == 1]
            n_samples = min(scaled_counts[i], len(subset_df))

            if n_samples > 0:
                chosen_df, remaining_df, picked_images_len = self.sample_patients(
                    n_samples, subset_df, remaining_df
                )

                if chosen_df is not None and len(chosen_df) > 0:
                    picked_counts = chosen_df[top_labels].sum(axis=0).to_numpy()
                    scaled_counts = np.maximum(0, scaled_counts - picked_counts)
                    frames.append(chosen_df)

        if len(frames) == 0:
            print('No data could be found for the chosen top labels.')
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
                    need, pos_pool, remaining_df
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
                    need, neg_pool, remaining_df
                )

                if chosen_df is not None and len(chosen_df) > 0:
                    df_representing = pd.concat(
                        [df_representing, chosen_df],
                        axis=0
                    )
                    need -= picked_images_len

        self.data_df = df_representing.drop_duplicates()

    @staticmethod
    def sample_patients(n_samples, subset_df, remaining_df):
        if n_samples <= 0 or len(subset_df) == 0:
            return None, remaining_df, 0

        candidate_pids = subset_df["Patient ID"].drop_duplicates()
        candidate_pids = candidate_pids.sample(frac=1.0).to_numpy()

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
