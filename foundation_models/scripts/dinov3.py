import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CHMv2ForDepthEstimation, CHMv2ImageProcessor

# Anchor to THIS file's directory, not the current working directory.
# This makes the script runnable from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent


def estimate_depth(image_path):
    """Run DINOv3 depth estimation and return (PIL image, depth tensor)."""
    processor = CHMv2ImageProcessor.from_pretrained(
        "facebook/dinov3-vitl16-chmv2-dpt-head")
    model = CHMv2ForDepthEstimation.from_pretrained(
        "facebook/dinov3-vitl16-chmv2-dpt-head")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"]

    return image, depth


def save_depth_map(depth, out_path):
    """Normalize the depth tensor to 0-255 and save as a grayscale PNG."""
    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_img = Image.fromarray((depth * 255).astype(np.uint8))
    depth_img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="DINOv3 depth estimation")
    # Default to the bundled image, resolved relative to the script.
    parser.add_argument(
        "image",
        nargs="?",
        default=str(SCRIPT_DIR / "church_00073.png"),
        help="Path to the input image (defaults to the bundled church image).",
    )
    parser.add_argument(
        "-o", "--output",
        default=str(SCRIPT_DIR / "depth_out.png"),
        help="Where to save the depth map PNG.",
    )
    args = parser.parse_args()

    image, depth = estimate_depth(args.image)
    print(f"Input:  {args.image}  ({image.width}x{image.height})")
    print(f"Depth:  shape={tuple(depth.shape)}  "
          f"min={depth.min():.3f}  max={depth.max():.3f}")

    out = save_depth_map(depth, args.output)
    print(f"Saved depth map -> {out}")


if __name__ == "__main__":
    main()
