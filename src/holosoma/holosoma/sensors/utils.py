import numpy as np


def _prepare_depth_for_visualization(
    depth: np.ndarray, near_clip: float, far_clip: float
) -> np.ndarray:
    """Prepare depth frame for visualization by clipping and scaling."""
    # clip and scale to [0, 1]
    depth = np.clip(depth, near_clip, far_clip)
    depth = (depth - near_clip) / (far_clip - near_clip)

    # [0, 1] -> [0, 255]
    return (depth * 255.0).astype(np.uint8)
