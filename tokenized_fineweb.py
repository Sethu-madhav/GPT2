import os
import shutil
from pathlib import Path
from tqdm import tqdm

from huggingface_hub import snapshot_download

TARGET_DIR = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")
REPO_ID = "ShallowU/FineWeb-Edu-10B-Tokens-NPY"  # dataset repo with pre-tokenized .npy files

os.makedirs(TARGET_DIR, exist_ok=True)

# Download only .npy (and optionally .npz) files from the dataset repo
snapshot_path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=["**/*.npy", "**/*.npz"],
    resume_download=True,
)

files = list(Path(snapshot_path).rglob("*.npy"))
if not files:
    raise RuntimeError(f"No .npy files found in snapshot at {snapshot_path} for {REPO_ID}")

for src in tqdm(files, desc="Copying .npy files to edu_fineweb10B", unit="file"):
    dst = Path(TARGET_DIR) / src.name
    if dst.exists():
        continue
    shutil.copy2(src, dst)

print(f"Done. Copied {len(files)} files to {TARGET_DIR}")