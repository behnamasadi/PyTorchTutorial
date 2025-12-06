import yaml
from utility.file_utils import resource_path
import pathlib
import os

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    kagglehub = None


if not KAGGLEHUB_AVAILABLE:
    raise ImportError(
        "kagglehub is required for automatic dataset download. "
        "Install it with: pip install kagglehub"
    )
else:
    print(f"kagglehub is available: {kagglehub.__version__}")


config_path = resource_path("../config/train.yaml")

config_path_absolute = pathlib.Path(config_path).resolve()
if config_path_absolute.exists():
    print(f"Reading config from{config_path_absolute}")
else:
    print(f"No config found at {config_path_absolute}")
    exit()


with open(config_path_absolute) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)


print(cfg["data"]["kaggle_dataset"])
print(cfg["data"]["path"])


"""
Download the dataset from Kaggle if it doesn't already exist.
Checks if train and val directories exist and have content before downloading.

Uses KAGGLE_USERNAME and KAGGLE_KEY environment variables for authentication.
"""


try:
    print(f"\nAttempting to download dataset: {cfg['data']['kaggle_dataset']}")
    dataset_path = kagglehub.dataset_download(cfg["data"]["kaggle_dataset"])

    print(f"\n{'='*70}")
    print(f"✅ Dataset downloaded successfully")
    print(f"{'='*70}")
    print(f"  Source (Kaggle cache): {dataset_path}")

    # Create symlink from kaggle_structure data directory to Kaggle cache
    # Resolve path relative to kaggle_structure directory (parent of config directory)
    kaggle_structure_root = config_path_absolute.parent.parent
    target_path = (kaggle_structure_root / cfg["data"]["path"]).resolve()
    source_path = pathlib.Path(dataset_path)

    # Remove existing directory/symlink if it exists
    if target_path.exists() or target_path.is_symlink():
        if target_path.is_symlink():
            target_path.unlink()
        elif target_path.is_dir():
            print(f"  ⚠️  Target directory exists, skipping symlink creation")
            print(f"  Target (Project dir):  {target_path}")
            print(f"{'='*70}")
        else:
            target_path.unlink()

    if not target_path.exists():
        # Create parent directories if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Create symlink
        target_path.symlink_to(source_path)
        print(f"  Target (Project dir):  {target_path}")
        print(f"{'='*70}")
        print(f"\n✅ Symlink created successfully")
        print(f"   {target_path} -> {source_path}")
        print(f"   (no disk space wasted)")
        print(f"{'='*70}")

except Exception as e:
    error_msg = (
        f"Failed to download or organize dataset: {e}\n"
        f"\nAuthentication options:\n"
        f"  - RunPod/Docker: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
        f"  - Local: Use kagglehub.login() or create ~/.kaggle/kaggle.json\n"
        f"  - Or ensure dataset is already downloaded to: {cfg['data']['path']}"
    )
    raise RuntimeError(error_msg) from e
