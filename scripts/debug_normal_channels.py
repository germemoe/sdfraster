"""Debug normal map channel flipping.

Try all possible channel flip combinations to find the correct mapping
between rendered normal and depth-derived normal.

Usage:
    python scripts/debug_normal_channels.py <input_image> [output_dir]
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import itertools


def extract_normal_cells(img):
    """Extract the 3 normal images from the 2x3 grid."""
    H, W = img.shape[:2]
    cell_h = H // 2
    cell_w = W // 3

    # Bottom row: Normal, Depth->N, DepthMed->N
    normal_rendered = img[cell_h:, :cell_w]           # 左下: 渲染的Normal
    normal_depth = img[cell_h:, cell_w:2*cell_w]      # 中下: Depth->Normal
    normal_depth_med = img[cell_h:, 2*cell_w:]        # 右下: DepthMed->Normal

    return normal_rendered, normal_depth, normal_depth_med


def vis_to_normal(vis):
    """Convert visualization [0,255] back to normal [-1,1].

    MILo convention: vis = (1 - normal) / 2 * 255
    So: normal = 1 - 2 * (vis / 255)
    """
    return 1.0 - 2.0 * (vis.astype(np.float32) / 255.0)


def normal_to_vis(normal):
    """Convert normal [-1,1] to visualization [0,255].

    MILo convention: vis = (1 - normal) / 2 * 255
    """
    vis = (1.0 - normal) / 2.0 * 255.0
    return np.clip(vis, 0, 255).astype(np.uint8)


def flip_channels(normal, flip_x, flip_y, flip_z):
    """Flip specified channels of normal map."""
    result = normal.copy()
    if flip_x:
        result[..., 0] = -result[..., 0]
    if flip_y:
        result[..., 1] = -result[..., 1]
    if flip_z:
        result[..., 2] = -result[..., 2]
    return result


def swap_channels(normal, order):
    """Swap channels according to order (e.g., [0,1,2], [1,0,2], etc.)."""
    return normal[..., order]


def main(input_path, output_dir=None):
    if output_dir is None:
        output_dir = Path(input_path).parent / "normal_debug"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Read image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Cannot read image from {input_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract normal cells
    normal_rendered, normal_depth, normal_depth_med = extract_normal_cells(img)

    # Skip label area (top ~30 pixels)
    label_h = 30
    normal_rendered = normal_rendered[label_h:]
    normal_depth = normal_depth[label_h:]
    normal_depth_med = normal_depth_med[label_h:]

    # Convert to normal space [-1, 1]
    n_rendered = vis_to_normal(normal_rendered)
    n_depth = vis_to_normal(normal_depth)

    print(f"Rendered normal stats: min={n_rendered.min():.3f}, max={n_rendered.max():.3f}")
    print(f"Depth normal stats: min={n_depth.min():.3f}, max={n_depth.max():.3f}")

    # Try all flip combinations for rendered normal
    print("\n=== Flip combinations ===")
    flip_combinations = list(itertools.product([False, True], repeat=3))

    results = []
    for i, (flip_x, flip_y, flip_z) in enumerate(flip_combinations):
        n_flipped = flip_channels(n_rendered, flip_x, flip_y, flip_z)
        vis_flipped = normal_to_vis(n_flipped)

        # Compute difference with depth normal
        diff = np.abs(vis_flipped.astype(float) - normal_depth.astype(float)).mean()

        flip_str = f"{'X' if flip_x else '_'}{'Y' if flip_y else '_'}{'Z' if flip_z else '_'}"
        results.append((flip_str, diff, vis_flipped))
        print(f"  Flip {flip_str}: avg_diff = {diff:.2f}")

    # Sort by difference
    results.sort(key=lambda x: x[1])

    print(f"\n=== Best matches ===")
    for flip_str, diff, _ in results[:3]:
        print(f"  Flip {flip_str}: avg_diff = {diff:.2f}")

    # Also try channel swaps
    print("\n=== Channel swap combinations ===")
    swap_orders = [
        [0, 1, 2],  # RGB
        [0, 2, 1],  # RBG
        [1, 0, 2],  # GRB
        [1, 2, 0],  # GBR
        [2, 0, 1],  # BRG
        [2, 1, 0],  # BGR
    ]
    swap_names = ['RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR']

    swap_results = []
    for order, name in zip(swap_orders, swap_names):
        n_swapped = swap_channels(n_rendered, order)
        vis_swapped = normal_to_vis(n_swapped)
        diff = np.abs(vis_swapped.astype(float) - normal_depth.astype(float)).mean()
        swap_results.append((name, diff, vis_swapped))
        print(f"  Swap {name}: avg_diff = {diff:.2f}")

    # Try swap + flip combinations
    print("\n=== Swap + Flip combinations (top 10) ===")
    all_results = []
    for order, swap_name in zip(swap_orders, swap_names):
        n_swapped = swap_channels(n_rendered, order)
        for flip_x, flip_y, flip_z in flip_combinations:
            n_flipped = flip_channels(n_swapped, flip_x, flip_y, flip_z)
            vis_flipped = normal_to_vis(n_flipped)
            diff = np.abs(vis_flipped.astype(float) - normal_depth.astype(float)).mean()
            flip_str = f"{'X' if flip_x else '_'}{'Y' if flip_y else '_'}{'Z' if flip_z else '_'}"
            combo_name = f"{swap_name}_{flip_str}"
            all_results.append((combo_name, diff, vis_flipped))

    all_results.sort(key=lambda x: x[1])
    for combo_name, diff, _ in all_results[:10]:
        print(f"  {combo_name}: avg_diff = {diff:.2f}")

    # Save comparison images
    print(f"\n=== Saving results to {output_dir} ===")

    # Save original comparison
    H, W = normal_rendered.shape[:2]
    comparison = np.zeros((H, W * 3, 3), dtype=np.uint8)
    comparison[:, :W] = normal_rendered
    comparison[:, W:2*W] = normal_depth
    comparison[:, 2*W:] = normal_depth_med
    cv2.imwrite(str(output_dir / "00_original.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    # Save top 10 best matches
    for i, (combo_name, diff, vis) in enumerate(all_results[:10]):
        comparison = np.zeros((H, W * 3, 3), dtype=np.uint8)
        comparison[:, :W] = vis                    # Modified rendered
        comparison[:, W:2*W] = normal_depth        # Target (depth normal)
        comparison[:, 2*W:] = np.abs(vis.astype(np.int16) - normal_depth.astype(np.int16)).astype(np.uint8) * 3  # Diff

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, f'{combo_name} (diff={diff:.1f})', (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, 'Depth->N', (W + 10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, 'Difference x3', (2*W + 10, 25), font, 0.6, (255, 255, 255), 2)

        cv2.imwrite(str(output_dir / f"{i+1:02d}_{combo_name}.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print(f"Done! Check {output_dir} for results.")
    print(f"Best match: {all_results[0][0]} with avg_diff = {all_results[0][1]:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_normal_channels.py <input_image> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    main(input_path, output_dir)
