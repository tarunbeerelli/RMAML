"""
RMAML training entry point.

Local dev (synthetic data, CPU):
    poetry run python train.py --config configs/conv4_miniimagenet.yaml --smoke-test

Full training (real data, GPU):
    python train.py --config configs/conv4_miniimagenet.yaml
"""

import argparse
import logging
import os

import torch
import yaml
from rmaml.datasets.episode_sampler import EpisodeSampler
from rmaml.datasets.synthetic import make_synthetic_dataset
from rmaml.meta_learner import RMAMLTrainer
from rmaml.models.rmaml_model import RMAMLModel
from rmaml.utils.tracking import RunTracker, setup_tracking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_sampler(cfg: dict, smoke_test: bool) -> EpisodeSampler:
    """
    Build episode sampler.
    smoke_test=True  → synthetic data (no disk needed, fast)
    smoke_test=False → real MiniImageNet from cfg['data']['root']
    """
    if smoke_test:
        log.info("Using synthetic dataset (smoke test mode)")
        dataset_map = make_synthetic_dataset(n_classes=64, images_per_class=20)
    else:
        # Real dataset loading goes here in Phase 5
        # For now raise a clear error
        raise NotImplementedError(
            "Real dataset loading not implemented yet. "
            "Run with --smoke-test for now."
        )

    return EpisodeSampler(
        dataset_map,
        n_way=cfg["model"]["n_way"],
        k_shot=cfg["data"]["k_shot"],
        q_query=cfg["data"]["q_query"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/conv4_miniimagenet.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 10 steps with synthetic data to verify pipeline",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n_epochs = 10 if args.smoke_test else cfg["meta"]["n_epochs"]

    device = get_device()
    log.info(f"Device: {device}")

    # Build components
    model = RMAMLModel(n_way=cfg["model"]["n_way"])
    trainer = RMAMLTrainer(
        model=model,
        n_inner=cfg["meta"]["n_inner"],
        alpha=cfg["meta"]["alpha"],
        outer_lr=cfg["optimizer"]["outer_lr"],
        meta_batch=cfg["meta"]["meta_batch"],
        device=device,
        use_cadam=(cfg["optimizer"]["type"] == "cadam"),
    )
    sampler = build_sampler(cfg, smoke_test=args.smoke_test)

    # Create checkpoint dir
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop with MLflow tracking
    setup_tracking(cfg["experiment"])
    run_name = "smoke-test" if args.smoke_test else cfg["optimizer"]["type"]

    with RunTracker(cfg, run_name=run_name) as tracker:
        for epoch in range(n_epochs):
            episodes = sampler.sample_batch(cfg["meta"]["meta_batch"])
            metrics = trainer.train_step(episodes)

            # Log metrics
            if epoch % cfg["logging"]["log_every"] == 0 or args.smoke_test:
                log.info(f"Epoch {epoch:>6} | loss={metrics['meta_loss']:.4f}")
                tracker.log(step=epoch, **metrics)

            # Orthogonality check
            if epoch % cfg["logging"]["orth_check_every"] == 0:
                err = model.orthogonality_error()
                log.info(f"           | orth_error={err:.2e}")
                tracker.log(step=epoch, orth_error=err)

            # Evaluation
            if epoch % cfg["logging"]["eval_every"] == 0 and not args.smoke_test:
                eval_eps = sampler.sample_batch(32)
                eval_metrics = trainer.evaluate(eval_eps)
                log.info(f"           | acc={eval_metrics['accuracy']:.4f}")
                tracker.log(step=epoch, **eval_metrics)

            # Checkpoint
            if epoch % cfg["logging"]["checkpoint_every"] == 0 and not args.smoke_test:
                path = f"checkpoints/epoch_{epoch:06d}.pt"
                torch.save(model.state_dict(), path)
                tracker.log_artifact(path)
                log.info(f"           | saved {path}")

    log.info("Training complete.")


if __name__ == "__main__":
    main()
