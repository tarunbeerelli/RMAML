"""
Tests for the RMAML training loop.
Uses synthetic data and CPU — no GPU or real dataset needed.
"""

import torch
from rmaml.datasets.episode_sampler import EpisodeSampler
from rmaml.datasets.synthetic import make_synthetic_dataset
from rmaml.meta_learner import RMAMLTrainer, inner_step
from rmaml.models.rmaml_model import RMAMLModel

DEVICE = torch.device("cpu")


def make_trainer(use_cadam: bool = True) -> tuple[RMAMLTrainer, EpisodeSampler]:
    model = RMAMLModel(n_way=5)
    trainer = RMAMLTrainer(
        model=model,
        n_inner=2,  # fewer steps to keep tests fast
        alpha=0.1,
        outer_lr=1e-3,
        meta_batch=2,
        device=DEVICE,
        use_cadam=use_cadam,
    )
    dataset_map = make_synthetic_dataset(n_classes=20, images_per_class=20)
    sampler = EpisodeSampler(dataset_map, n_way=5, k_shot=1, q_query=15)
    return trainer, sampler


def test_inner_step_returns_updated_params():
    """Inner step should return a param dict with same keys."""
    model = RMAMLModel(n_way=5)
    params = {n: p.clone() for n, p in model.named_parameters()}
    dataset_map = make_synthetic_dataset(n_classes=20, images_per_class=20)
    sampler = EpisodeSampler(dataset_map, n_way=5, k_shot=1, q_query=15)
    ep = sampler.sample()

    new_params = inner_step(params, model, ep, alpha=0.1, device=DEVICE)
    assert set(new_params.keys()) == set(params.keys())


def test_stiefel_param_stays_on_manifold():
    """Classifier weight should remain orthogonal after inner step."""
    model = RMAMLModel(n_way=5)
    params = {n: p.clone() for n, p in model.named_parameters()}
    dataset_map = make_synthetic_dataset(n_classes=20, images_per_class=20)
    sampler = EpisodeSampler(dataset_map, n_way=5, k_shot=1, q_query=15)
    ep = sampler.sample()

    new_params = inner_step(params, model, ep, alpha=0.1, device=DEVICE)
    W = new_params["classifier.weight"]
    err = (W.T @ W - torch.eye(5)).norm().item()
    assert err < 1e-4, f"Stiefel constraint violated after inner step: {err:.2e}"


def test_train_step_returns_loss():
    """train_step should return a finite loss."""
    trainer, sampler = make_trainer(use_cadam=True)
    episodes = sampler.sample_batch(2)
    metrics = trainer.train_step(episodes)

    assert "meta_loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["meta_loss"]))


def test_loss_finite_over_steps():
    """Loss should remain finite across multiple training steps."""
    trainer, sampler = make_trainer(use_cadam=True)
    for _ in range(5):
        eps = sampler.sample_batch(2)
        m = trainer.train_step(eps)
        assert torch.isfinite(
            torch.tensor(m["meta_loss"])
        ), f"Loss became non-finite: {m['meta_loss']}"


def test_cadam_vs_csgdm_both_run():
    """Both optimizers should run without error."""
    for use_cadam in [True, False]:
        trainer, sampler = make_trainer(use_cadam=use_cadam)
        eps = sampler.sample_batch(2)
        metrics = trainer.train_step(eps)
        assert torch.isfinite(torch.tensor(metrics["meta_loss"]))


def test_evaluate_returns_accuracy():
    """evaluate should return accuracy between 0 and 1."""
    trainer, sampler = make_trainer()
    eps = sampler.sample_batch(4)
    metrics = trainer.evaluate(eps)

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
