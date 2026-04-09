from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

DEFAULT_DISEASES = [
    "Negative",
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax",
    "Consolidation",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Emphysema",
    "Edema",
    "Fibrosis",
    "Pneumonia",
    "Hernia",
]


def _infer_project_root():
    """
    Try to infer the project root from this file location.

    Expected repo structure:
    NIH_dataset_nn/
    ├── Cached_Data/cache_224_tot_90_jpg
    ├── Information/Data_Entry_2017.csv
    └── NIH_Code/
    """
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "Information" / "Data_Entry_2017.csv").exists():
            return parent
    raise FileNotFoundError(
        "Could not infer project root. Please place this file inside the project "
        "or edit _infer_project_root()."
    )


def _load_metadata(label_file):
    df = pd.read_csv(label_file).copy()

    if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
        raise ValueError(
            "CSV must contain at least 'Image Index' and 'Finding Labels' columns."
        )

    df["Finding Labels"] = df["Finding Labels"].fillna("").astype(str)

    def normalize_label_string(label_string):
        labels = [x.strip() for x in label_string.split("|") if x.strip()]
        labels = ["Negative" if x == "No Finding" else x for x in labels]
        return "|".join(labels)

    df["Finding Labels"] = df["Finding Labels"].apply(normalize_label_string)
    return df


def _get_image_path(image_name, original_root=None, cache_dir=None, use_cache=False):
    image_name = str(image_name)

    if use_cache:
        if cache_dir is None:
            raise ValueError("cache_dir is required when use_cache=True.")

        cache_dir = Path(cache_dir)
        stem = Path(image_name).stem
        jpg_path = cache_dir / f"{stem}.jpg"
        jpeg_path = cache_dir / f"{stem}.jpeg"

        if jpg_path.exists():
            return jpg_path
        if jpeg_path.exists():
            return jpeg_path

        raise FileNotFoundError(f"Cached image not found for: {image_name}")

    if original_root is None:
        raise ValueError("original_root is required when use_cache=False.")

    original_root = Path(original_root)

    # Most robust recursive search.
    matches = list(original_root.rglob(image_name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Original image not found for: {image_name}")


def _label_set(label_string):
    if not label_string:
        return set()
    return {x.strip() for x in label_string.split("|") if x.strip()}


def _is_pure_match(label_string, disease):
    labels = _label_set(label_string)

    if disease == "Negative":
        return labels == {"Negative"}

    return labels == {disease}


def _is_mixed_match(label_string, disease):
    labels = _label_set(label_string)

    if disease == "Negative":
        return False

    return disease in labels and len(labels) > 1


def _prepare_candidates(df, disease):
    disease = str(disease).strip()

    if disease.lower() == "no finding":
        disease = "Negative"

    valid_diseases = set(DEFAULT_DISEASES)
    if disease not in valid_diseases:
        raise ValueError(
            f"Unknown disease '{disease}'. Valid values are: {sorted(valid_diseases)}"
        )

    pure_df = df[df["Finding Labels"].apply(lambda x: _is_pure_match(x, disease))].copy()
    mixed_df = df[df["Finding Labels"].apply(lambda x: _is_mixed_match(x, disease))].copy()

    return disease, pure_df, mixed_df


def _resolve_grid_slots(grid_size):
    """
    Allowed values:
    - 1  -> 1 slot
    - 2  -> 2 slots
    - 4  -> 4 slots (2x2)
    """
    if grid_size not in (1, 2, 4):
        raise ValueError("grid_size must be one of: 1, 2, 4")
    return grid_size


def _sample_rows(pure_df, mixed_df, total_needed, random_state=42):
    pure_take = min(len(pure_df), total_needed)
    pure_sample = (
        pure_df.sample(n=pure_take, random_state=random_state)
        if pure_take > 0
        else pure_df.iloc[0:0].copy()
    )

    remaining = total_needed - pure_take

    mixed_sample = mixed_df.iloc[0:0].copy()
    if remaining > 0 and len(mixed_df) > 0:
        mixed_take = min(len(mixed_df), remaining)
        mixed_sample = mixed_df.sample(n=mixed_take, random_state=random_state)

    selected = pd.concat([pure_sample, mixed_sample], axis=0).reset_index(drop=True)
    return selected, pure_take, len(mixed_sample)


def _make_figure_layout(n_images):
    if n_images <= 0:
        raise ValueError("n_images must be >= 1")

    if n_images == 1:
        return plt.subplots(1, 1, figsize=(8, 8))

    if n_images == 2:
        return plt.subplots(1, 2, figsize=(16, 8))

    if n_images == 3:
        return plt.subplots(2, 2, figsize=(16, 16))

    return plt.subplots(2, 2, figsize=(16, 16))


def load_disease_image_grid(
    disease,
    grid_size=4,
    use_cache=False,
    random_state=42,
    original_root=None,
    cache_dir=None,
    label_file=None,
    show_titles=True,
):
    """
    Load and display a disease image grid.

    Selection logic:
    1. First use PURE samples:
       - Negative -> label must be exactly 'Negative'
       - Any pathology -> label must be exactly that pathology only
    2. If pure samples are not enough, fill remaining slots with MIXED samples
       that still contain the requested pathology.
    3. If total found samples are fewer than requested, display a smaller grid.

    Parameters
    ----------
    disease : str
        Example: "Negative", "Effusion", "Hernia"
    grid_size : int
        One of: 1, 2, 4
    use_cache : bool
        False -> load original images
        True  -> load cached JPEG images
    random_state : int
        Sampling seed
    original_root : str or Path or None
        Folder containing original NIH images
    cache_dir : str or Path or None
        Cache folder
    label_file : str or Path or None
        Metadata CSV path
    show_titles : bool
        Show title above each image

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib axes
    selected_df : pandas.DataFrame
        Selected rows in display order
    """
    project_root = _infer_project_root()

    if label_file is None:
        label_file = project_root / "Information" / "Data_Entry_2017.csv"
    else:
        label_file = Path(label_file)

    if cache_dir is None:
        cache_dir = project_root / "Cached_Data" / "cache_224_tot_90_jpg"
    else:
        cache_dir = Path(cache_dir)

    if original_root is None:
        # Edit this if your original NIH images are stored elsewhere.
        # This function assumes you will provide original_root if needed.
        possible_original_dirs = [
            project_root / "images",
            project_root / "Images",
            project_root / "NIH_Images",
            project_root / "data",
        ]
        original_root = None
        for p in possible_original_dirs:
            if p.exists():
                original_root = p
                break

        if original_root is None and not use_cache:
            raise FileNotFoundError(
                "original_root was not provided and no default original image folder was found. "
                "Pass original_root=... explicitly for original images."
            )
    else:
        original_root = Path(original_root)

    requested_slots = _resolve_grid_slots(grid_size)

    df = _load_metadata(label_file)
    disease, pure_df, mixed_df = _prepare_candidates(df, disease)

    selected_df, n_pure, n_mixed = _sample_rows(
        pure_df=pure_df,
        mixed_df=mixed_df,
        total_needed=requested_slots,
        random_state=random_state,
    )

    n_found = len(selected_df)
    if n_found == 0:
        raise ValueError(f"No images found for disease '{disease}'.")

    fig, axes = _make_figure_layout(n_found)

    if n_found == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else list(axes)

    loaded_paths = []

    for i, (_, row) in enumerate(selected_df.iterrows()):
        image_name = row["Image Index"]
        label_string = row["Finding Labels"]

        image_path = _get_image_path(
            image_name=image_name,
            original_root=original_root,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
        loaded_paths.append(str(image_path))

        img = Image.open(image_path).convert("RGB")

        ax = axes_list[i]
        ax.imshow(img)
        ax.axis("off")

        if show_titles:
            sample_type = "PURE" if _is_pure_match(label_string, disease) else "MIXED"
            ax.set_title(
                f"{image_name}\n{sample_type}: {label_string}",
                fontsize=10,
            )

    # Hide any unused axes in the 2x2 case when only 3 images are shown.
    for j in range(n_found, len(axes_list)):
        axes_list[j].axis("off")

    source_text = "cache" if use_cache else "original"
    fig.suptitle(
        (
            f"{disease} samples | requested={requested_slots}, shown={n_found} | "
            f"pure={n_pure}, mixed={n_mixed} | source={source_text}"
        ),
        fontsize=14,
    )
    plt.tight_layout()

    selected_df = selected_df.copy()
    selected_df["resolved_image_path"] = loaded_paths
    selected_df["selection_type"] = selected_df["Finding Labels"].apply(
        lambda x: "PURE" if _is_pure_match(x, disease) else "MIXED"
    )

    return fig, axes, selected_df


def save_disease_image_grid(
    disease,
    grid_size=4,
    use_cache=False,
    random_state=42,
    original_root=None,
    cache_dir=None,
    label_file=None,
    output_dir=None,
    extension="png",
    dpi=150,
    show_titles=True,
    close_figure=True,
):
    """
    Create and save one disease grid.

    Output path:
    pathology_samples/<disease>.<extension>
    """
    fig, axes, selected_df = load_disease_image_grid(
        disease=disease,
        grid_size=grid_size,
        use_cache=use_cache,
        random_state=random_state,
        original_root=original_root,
        cache_dir=cache_dir,
        label_file=label_file,
        show_titles=show_titles,
    )

    project_root = _infer_project_root()

    if output_dir is None:
        output_dir = project_root / "pathology_samples"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = str(disease).replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"{safe_name}.{extension}"

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if close_figure:
        plt.close(fig)

    return output_path, selected_df


def save_all_disease_image_grids(
    diseases=None,
    grid_size=4,
    use_cache=False,
    random_state=42,
    original_root=None,
    cache_dir=None,
    label_file=None,
    output_dir=None,
    extension="png",
    dpi=150,
    show_titles=True,
):
    """
    Create and save one grid for each disease.

    Returns
    -------
    results : dict
        {
            disease_name: {
                "output_path": Path(...),
                "num_selected": int,
                "num_pure": int,
                "num_mixed": int,
            }
        }
    """
    if diseases is None:
        diseases = DEFAULT_DISEASES

    results = {}

    for disease in diseases:
        output_path, selected_df = save_disease_image_grid(
            disease=disease,
            grid_size=grid_size,
            use_cache=use_cache,
            random_state=random_state,
            original_root=original_root,
            cache_dir=cache_dir,
            label_file=label_file,
            output_dir=output_dir,
            extension=extension,
            dpi=dpi,
            show_titles=show_titles,
            close_figure=True,
        )

        n_pure = int((selected_df["selection_type"] == "PURE").sum())
        n_mixed = int((selected_df["selection_type"] == "MIXED").sum())

        results[disease] = {
            "output_path": output_path,
            "num_selected": len(selected_df),
            "num_pure": n_pure,
            "num_mixed": n_mixed,
        }

def save_all_disease_image_grids_to_pdf(
        diseases=None,
        grid_size=4,
        use_cache=False,
        random_state=42,
        original_root=None,
        cache_dir=None,
        label_file=None,
        output_path=None,
        dpi=150,
        show_titles=True,
):
    """
    Create one PDF containing all disease grids (one page per disease).
    """
    if diseases is None:
        diseases = DEFAULT_DISEASES

    project_root = _infer_project_root()

    if output_path is None:
        output_path = project_root / "pathology_samples" / "all_pathologies.pdf"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for disease in diseases:
            fig, axes, _ = load_disease_image_grid(
                disease=disease,
                grid_size=grid_size,
                use_cache=use_cache,
                random_state=random_state,
                original_root=original_root,
                cache_dir=cache_dir,
                label_file=label_file,
                show_titles=show_titles,
            )

            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    return output_path

    return results
if __name__ == "__main__":
    save_all_disease_image_grids()
    save_all_disease_image_grids_to_pdf()