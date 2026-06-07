"""
Episode sampler for N-way K-shot meta-learning.
Works with any class-indexed dataset dict — real or synthetic.
"""

import random
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Episode:
    """One few-shot episode: support set + query set."""

    support_x: Tensor  # (n_way * k_shot, C, H, W)
    support_y: Tensor  # (n_way * k_shot,)
    query_x: Tensor  # (n_way * q_query, C, H, W)
    query_y: Tensor  # (n_way * q_query,)
    n_way: int
    k_shot: int


class EpisodeSampler:
    """
    Samples N-way K-shot episodes from a class-indexed dataset.

    Args:
        dataset_map: dict mapping class_id (int) → list of image tensors
        n_way:       number of classes per episode
        k_shot:      support examples per class
        q_query:     query examples per class
    """

    def __init__(
        self,
        dataset_map: dict[int, list[Tensor]],
        n_way: int = 5,
        k_shot: int = 1,
        q_query: int = 15,
    ):
        self.dataset_map = dataset_map
        self.classes = list(dataset_map.keys())
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # Validate we have enough images per class
        needed = k_shot + q_query
        for cls, imgs in dataset_map.items():
            assert (
                len(imgs) >= needed
            ), f"Class {cls} has {len(imgs)} images, need at least {needed}"

    def sample(self) -> Episode:
        """Sample one episode."""
        chosen_classes = random.sample(self.classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for label, cls in enumerate(chosen_classes):
            imgs = random.sample(self.dataset_map[cls], self.k_shot + self.q_query)
            for img in imgs[: self.k_shot]:
                support_x.append(img)
                support_y.append(label)
            for img in imgs[self.k_shot :]:
                query_x.append(img)
                query_y.append(label)

        return Episode(
            support_x=torch.stack(support_x),
            support_y=torch.tensor(support_y, dtype=torch.long),
            query_x=torch.stack(query_x),
            query_y=torch.tensor(query_y, dtype=torch.long),
            n_way=self.n_way,
            k_shot=self.k_shot,
        )

    def sample_batch(self, batch_size: int) -> list[Episode]:
        """Sample a batch of episodes for one meta-update."""
        return [self.sample() for _ in range(batch_size)]
