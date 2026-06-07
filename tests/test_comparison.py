"""
Comparison tests: RMAML vs MAML, cAdam vs cSGDM.

These tests verify the four trainers all run correctly
and produce finite losses. The actual accuracy comparison
happens during real training runs logged to MLflow —
not in unit tests (which use random synthetic data).
"""

import pytest
import torch
from rmaml.baselines.maml import MAMLModel, MAMLTrainer
from rmaml.datasets.episode_sampler import EpisodeSampler
from rmaml.datasets.synthetic import make_synthetic_dataset
from rmaml.meta_learner import RMAMLTrainer
from rmaml.models.rmaml_model import RMAMLModel

DEVICE = torch.device("cpu")
N_STEPS = 3  # keep tests fast


@pytest.fixture
def sampler():
    dataset_map = make_synthetic_dataset(n_classes=20, images_per_class=20)
    return EpisodeSampler(dataset_map, n_way=5, k_shot=1, q_query=15)


# ── Instantiation ─────────────────────────────────────────────────


@pytest.mark.parametrize("use_cadam", [True, False])
def test_rmaml_instantiates(use_cadam):
    model = RMAMLModel(n_way=5)
    trainer = RMAMLTrainer(model=model, use_cadam=use_cadam, device=DEVICE)
    assert trainer is not None


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_maml_instantiates(optimizer):
    model = MAMLModel(n_way=5)
    trainer = MAMLTrainer(model=model, optimizer=optimizer, device=DEVICE)
    assert trainer is not None


# ── All four trainers produce finite loss ─────────────────────────


@pytest.mark.parametrize("use_cadam", [True, False], ids=["cAdam", "cSGDM"])
def test_rmaml_loss_finite(sampler, use_cadam):
    model = RMAMLModel(n_way=5)
    trainer = RMAMLTrainer(model=model, n_inner=2, use_cadam=use_cadam, device=DEVICE)
    eps = sampler.sample_batch(2)
    metrics = trainer.train_step(eps)
    assert torch.isfinite(torch.tensor(metrics["meta_loss"]))


@pytest.mark.parametrize("optimizer", ["adam", "sgd"], ids=["Adam", "SGD"])
def test_maml_loss_finite(sampler, optimizer):
    model = MAMLModel(n_way=5)
    trainer = MAMLTrainer(model=model, n_inner=2, optimizer=optimizer, device=DEVICE)
    eps = sampler.sample_batch(2)
    metrics = trainer.train_step(eps)
    assert torch.isfinite(torch.tensor(metrics["meta_loss"]))


# ── RMAML maintains Stiefel constraint, MAML does not ────────────


def test_rmaml_classifier_stays_orthogonal(sampler):
    """RMAML classifier weight should remain orthogonal after training."""
    model = RMAMLModel(n_way=5)
    trainer = RMAMLTrainer(model=model, n_inner=2, device=DEVICE)
    for _ in range(N_STEPS):
        trainer.train_step(sampler.sample_batch(2))
    err = model.orthogonality_error()
    assert err < 1e-3, f"Orthogonality error too large: {err:.2e}"


def test_maml_classifier_not_orthogonal(sampler):
    """
    MAML classifier weight should drift from orthogonality —
    confirming it has no Stiefel constraint.
    """
    model = MAMLModel(n_way=5)
    trainer = MAMLTrainer(model=model, n_inner=2, device=DEVICE)
    for _ in range(N_STEPS):
        trainer.train_step(sampler.sample_batch(2))
    W = model.classifier.weight
    err = (W @ W.T - torch.eye(5)).norm().item()
    # MAML has no orthogonality constraint — error should be > 0
    assert err > 0


# ── Both models evaluate without error ────────────────────────────


def test_rmaml_evaluate(sampler):
    model = RMAMLModel(n_way=5)
    trainer = RMAMLTrainer(model=model, n_inner=2, device=DEVICE)
    metrics = trainer.evaluate(sampler.sample_batch(4))
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_maml_evaluate(sampler):
    model = MAMLModel(n_way=5)
    trainer = MAMLTrainer(model=model, n_inner=2, device=DEVICE)
    metrics = trainer.evaluate(sampler.sample_batch(4))
    assert 0.0 <= metrics["accuracy"] <= 1.0
