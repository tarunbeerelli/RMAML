"""
Proper few-shot evaluation on held-out test classes.

Follows standard protocol from the paper:
- 600 test episodes
- 5-way 1-shot, 15 query images per class
- Test split only (20 unseen classes never seen during training)
- Reports mean accuracy + 95% confidence interval

Usage:
    # Evaluate RMAML checkpoint
    PYTHONPATH=src python evaluate.py \
        --checkpoint checkpoints/epoch_059000.pt \
        --config configs/conv4_miniimagenet.yaml

    # Evaluate MAML checkpoint
    PYTHONPATH=src python evaluate.py \
        --checkpoint checkpoints/epoch_059000.pt \
        --config configs/conv4_maml.yaml \
        --maml
"""

import argparse
import logging
import math

import torch
import yaml
from rmaml.baselines.maml import MAMLModel, MAMLTrainer
from rmaml.datasets.episode_sampler import EpisodeSampler
from rmaml.datasets.miniimagenet import load_miniimagenet
from rmaml.meta_learner import RMAMLTrainer
from rmaml.models.rmaml_model import RMAMLModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

N_EPISODES = 600  # standard evaluation protocol


def confidence_interval(accs: list[float]) -> float:
    """95% confidence interval: 1.96 * std / sqrt(n)."""
    n = len(accs)
    mean = sum(accs) / n
    variance = sum((a - mean) ** 2 for a in accs) / (n - 1)
    return 1.96 * math.sqrt(variance / n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/conv4_miniimagenet.yaml")
    parser.add_argument("--maml", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Mode: {'MAML' if args.maml else 'RMAML'}")
    log.info(f"Evaluating on {args.split} split, {args.n_episodes} episodes")

    # Load model from checkpoint
    if args.maml:
        model = MAMLModel(n_way=cfg["model"]["n_way"])
    else:
        model = RMAMLModel(n_way=cfg["model"]["n_way"])

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # Build trainer (no outer optimizer needed for eval)
    if args.maml:
        trainer = MAMLTrainer(
            model=model,
            n_inner=cfg["meta"]["n_inner"],
            alpha=cfg["meta"]["alpha"],
            device=device,
        )
    else:
        trainer = RMAMLTrainer(
            model=model,
            n_inner=cfg["meta"]["n_inner"],
            alpha=cfg["meta"]["alpha"],
            device=device,
            use_cadam=(cfg["optimizer"]["type"] == "cadam"),
        )

    # Load test split — unseen classes only
    log.info(f"Loading MiniImageNet {args.split} split...")
    dataset_map = load_miniimagenet(
        root=cfg["data"]["root"],
        split=args.split,
        augment=False,  # no augmentation at test time
    )
    sampler = EpisodeSampler(
        dataset_map,
        n_way=cfg["model"]["n_way"],
        k_shot=cfg["data"]["k_shot"],
        q_query=cfg["data"]["q_query"],
    )

    # Run evaluation episodes
    log.info(f"Running {args.n_episodes} test episodes...")
    accs = []
    for i in range(args.n_episodes):
        episode = sampler.sample()
        metrics = trainer.evaluate([episode])
        accs.append(metrics["accuracy"])

        if (i + 1) % 100 == 0:
            mean = sum(accs) / len(accs)
            log.info(f"  Episode {i+1}/{args.n_episodes} | running acc={mean:.4f}")

    # Final result
    mean_acc = sum(accs) / len(accs)
    ci = confidence_interval(accs)

    log.info("=" * 50)
    log.info(f"  {'MAML' if args.maml else 'RMAML'} {cfg['optimizer']['type']}")
    log.info(f"  Split:    {args.split}")
    log.info(f"  Episodes: {args.n_episodes}")
    log.info(f"  Accuracy: {mean_acc*100:.2f}% ± {ci*100:.2f}%")
    log.info("=" * 50)

    import mlflow
    from rmaml.utils.tracking import setup_tracking

    setup_tracking(cfg["experiment"])
    run_name = f"eval_{args.split}_{'maml' if args.maml else cfg['optimizer']['type']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "checkpoint": args.checkpoint,
                "split": args.split,
                "n_episodes": args.n_episodes,
                "mode": "maml" if args.maml else "rmaml",
                "optimizer": cfg["optimizer"]["type"],
            }
        )
        mlflow.log_metrics(
            {
                "test_accuracy": mean_acc,
                "test_accuracy_pct": mean_acc * 100,
                "confidence_interval": ci,
            }
        )
        log.info(f"Results logged to MLflow run: {run_name}")


if __name__ == "__main__":
    main()
