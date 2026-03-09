import os
from glob import glob
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


def convert_one(src_path: str, dst_dir: str, size: int, quality: int):
    """
    Opens PNG -> RGB -> resize -> save as JPG with same base name.
    """
    base = os.path.basename(src_path)          # 00000001_000.png
    name, _ = os.path.splitext(base)           # 00000001_000
    dst_path = os.path.join(dst_dir, f"{name}.jpg")

    if os.path.exists(dst_path):
        return

    img = Image.open(src_path).convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    img.save(dst_path, "JPEG", quality=quality, optimize=True)


def build_cache(src_root: str, cache_dir: str, size: int = 224, quality: int = 90,
                workers: int = 6, limit: int | None = 50000):
    os.makedirs(cache_dir, exist_ok=True)

    pattern = os.path.join(src_root, "images_*", "images", "*.png")
    files = sorted(glob(pattern))
    if not files:
        raise RuntimeError(f"No image found with pattern: {pattern}")

    if limit is not None:
        files = files[:limit]

    fn = partial(convert_one, dst_dir=cache_dir, size=size, quality=quality)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(
                ex.map(fn, files),
                total=len(files),
                desc=f"Building cache {size}x{size} JPG"
        ):
            pass

    print(f" Cache created: {cache_dir}")
    print(f"Images processed: {len(files)}")

from pathlib import Path
if __name__ == "__main__":   
    BASE_DIR = Path(__file__).resolve().parent.parent  # parent of NIH_Code
    src_root = BASE_DIR / "Data" / "archive"
    cache_dir = BASE_DIR / "Cached_Data" / "cache_224_tot_90_jpg"
    build_cache(
        src_root=str(src_root),
        cache_dir=str(cache_dir),
        size=224,
        quality=90,
        workers=6,
        limit=None
    )