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
    raise ImportError("kagglehub is required. Install: pip install kagglehub")
else:
    print(f"kagglehub is available: {kagglehub.__version__}")

config_path = resource_path("../config/train.yaml")
config_path_absolute = pathlib.Path(config_path).resolve()
if not config_path_absolute.exists():
    print(f"No config found at {config_path_absolute}")
    exit()

print(f"Reading config from {config_path_absolute}")
with open(config_path_absolute) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

try:
    print(f"\nAttempting to download dataset: {cfg['data']['kaggle_dataset']}")
    dataset_path = kagglehub.dataset_download(cfg["data"]["kaggle_dataset"])
    print(f"\n{'='*70}")
    print(f"Dataset downloaded successfully")
    print(f"{'='*70}")
    print(f"  Source (Kaggle cache): {dataset_path}")

    kaggle_structure_root = config_path_absolute.parent.parent
    target_path = (kaggle_structure_root / cfg["data"]["path"]).resolve()
    source_path = pathlib.Path(dataset_path)

    if target_path.exists() or target_path.is_symlink():
        if target_path.is_symlink():
            target_path.unlink()
        elif target_path.is_dir():
            print(f"  Note: Target directory exists, skipping symlink creation")
        else:
            target_path.unlink()

    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.symlink_to(source_path)
        print(f"  Target (Project dir): {target_path}")
        print(f"  Symlink created: {target_path} -> {source_path}")
        print(f"{'='*70}")

except Exception as e:
    raise RuntimeError(
        f"Failed to download: {e}\n"
        f"Auth: RunPod/Docker: KAGGLE_USERNAME, KAGGLE_KEY | Local: ~/.kaggle/kaggle.json"
    ) from e



import os
import warnings

warnings.filterwarnings('ignore')
import kagglehub

print("\nSTEP 1: Downloading dataset from KaggleHub...")
try:
    path = kagglehub.dataset_download("orvile/brain-cancer-mri-dataset")
    print("Path to dataset files:", path)
    data_path = path
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Using local data path...")
    data_path = "data/brain-cancer/"

if not os.path.exists(data_path):
    print(f"Error: Data path '{data_path}' does not exist!")
    print("Please ensure the dataset is downloaded or available at the specified path.")
