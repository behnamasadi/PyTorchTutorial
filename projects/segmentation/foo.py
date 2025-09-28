import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.file_utils import resource_path
from torchvision import transforms
from typing import List, Tuple, Callable, Optional

from pathlib import Path


project_root = resource_path("")
project_root = Path(project_root)

data_path: str = "data/KITI/data_semantics/training"


root = project_root.joinpath(data_path)

image_dir: str = "image_2"
label_dir: str = "semantic"

img_root = root.joinpath(image_dir)
lbl_root = root.joinpath(label_dir)

print(Path.exists(img_root))
print(img_root)
print(lbl_root)

samples = []

for img_file_path in sorted(img_root.rglob("*.png")):

    img_file_name = img_file_path.relative_to(img_root)
    lbl_file_path = lbl_root.joinpath(img_file_name)
    if (lbl_file_path.exists()):
        # print("img_file_name", img_file_name)
        lbl_file_name = lbl_file_path.relative_to(lbl_root)
        # print("lbl_file_name:", lbl_file_name)
        samples.append((img_file_path, lbl_file_path))

print(samples[0])

exit()

# # Pair PNGs by stem
# samples: List[Tuple[Path, Path]] = []
# for img_path in sorted(self.img_root.rglob("*.png")):
#     rel = img_path.relative_to(self.img_root)
#     lbl_path = (self.lbl_root / rel).with_suffix(".png")
#     if lbl_path.exists():
#         self.samples.append((img_path, lbl_path))
# if not self.samples:
#     raise RuntimeError(
#         f"No PNG pairs under {self.img_root} and {self.lbl_root}")
