"""
Tests for episode sampler and synthetic dataset.
All run on CPU with no real data — safe for CI.
"""

import pytest
from rmaml.datasets.episode_sampler import EpisodeSampler
from rmaml.datasets.synthetic import make_synthetic_dataset


@pytest.fixture
def sampler():
    dataset_map = make_synthetic_dataset(n_classes=64, images_per_class=20)
    return EpisodeSampler(dataset_map, n_way=5, k_shot=1, q_query=15)


def test_synthetic_dataset_shape():
    """Dataset map has correct structure and image shapes."""
    dataset_map = make_synthetic_dataset(n_classes=64, images_per_class=20)
    assert len(dataset_map) == 64
    assert len(dataset_map[0]) == 20
    assert dataset_map[0][0].shape == (3, 84, 84)


def test_episode_support_shape(sampler):
    """Support set shape: (n_way * k_shot, C, H, W)."""
    ep = sampler.sample()
    assert ep.support_x.shape == (5, 3, 84, 84)  # 5-way 1-shot
    assert ep.support_y.shape == (5,)


def test_episode_query_shape(sampler):
    """Query set shape: (n_way * q_query, C, H, W)."""
    ep = sampler.sample()
    assert ep.query_x.shape == (75, 3, 84, 84)  # 5 * 15
    assert ep.query_y.shape == (75,)


def test_episode_labels_range(sampler):
    """Labels should be in range [0, n_way)."""
    ep = sampler.sample()
    assert ep.support_y.min() >= 0
    assert ep.support_y.max() < 5
    assert ep.query_y.min() >= 0
    assert ep.query_y.max() < 5


def test_batch_size(sampler):
    """sample_batch returns correct number of episodes."""
    batch = sampler.sample_batch(4)
    assert len(batch) == 4


def test_not_enough_images_raises():
    """Should raise if class has fewer images than k_shot + q_query."""
    dataset_map = make_synthetic_dataset(
        n_classes=10,
        images_per_class=5,  # only 5, need 16
    )
    with pytest.raises(AssertionError):
        EpisodeSampler(dataset_map, n_way=5, k_shot=1, q_query=15)
