"""Convert normal visualization from (normal+1)/2 to (1-normal)/2.

The relationship is:
  vis_new = (1 - normal) / 2 = 1 - (normal + 1) / 2 = 1 - vis_old

So for the normal images (bottom row), we just need to do: 255 - pixel_value

Usage:
    python scripts/convert_normal_vis.py <input_image> [output_image]
"""

import sys
import cv2
import numpy as np

def convert_normal_visualization(input_path, output_path=None):
    """Convert normal visualization from old to new convention."""
    if output_path is None:
        output_path = input_path.replace('.png', '_converted.png')

    # Read image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Cannot read image from {input_path}")
        return

    H, W = img.shape[:2]

    # The image is a 2x3 grid
    # Top row: RGB, Depth, Depth Med (don't change)
    # Bottom row: Normal, Depth->N, DepthMed->N (need to invert)

    cell_h = H // 2
    cell_w = W // 3

    # Invert the bottom row (normal visualizations)
    # Skip the label area (top ~30 pixels of each cell)
    label_height = 30

    # Bottom-left: Normal
    img[cell_h + label_height:, :cell_w] = 255 - img[cell_h + label_height:, :cell_w]

    # Bottom-middle: Depth->N
    img[cell_h + label_height:, cell_w:2*cell_w] = 255 - img[cell_h + label_height:, cell_w:2*cell_w]

    # Bottom-right: DepthMed->N
    img[cell_h + label_height:, 2*cell_w:] = 255 - img[cell_h + label_height:, 2*cell_w:]

    # Save result
    cv2.imwrite(output_path, img)
    print(f"Converted image saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_normal_vis.py <input_image> [output_image]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_normal_visualization(input_path, output_path)
