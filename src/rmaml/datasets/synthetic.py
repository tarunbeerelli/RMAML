"""
Synthetic dataset for local development and CI.
Generates random tensors with the same shape as MiniImageNet (3, 84, 84).
No real data needed — just verifies shapes and logic are correct.
"""

import torch


def make_synthetic_dataset(
    n_classes: int = 64,
    images_per_class: int = 20,
    image_size: int = 84,
    channels: int = 3,
) -> dict[int, list[torch.Tensor]]:
    """
    Returns a dataset_map matching the EpisodeSampler interface.
    Images are random tensors normalised to [0, 1].
    """
    return {
        cls: [
            torch.rand(channels, image_size, image_size)
            for _ in range(images_per_class)
        ]
        for cls in range(n_classes)
    }
