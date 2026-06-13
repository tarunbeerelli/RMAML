"""
Ray Tune HPO for RMAML.
Searches over inner LR, outer LR, n_inner steps, and optimizer type.
Uses ASHA scheduler to kill bad trials early.

Run:
    PYTHONPATH=src python tune.py --config configs/conv4_miniimagenet.yaml

Results saved to:
    ray_results/rmaml_hpo/   — trial checkpoints and metrics
    mlruns/                  — best config logged to MLflow
"""

import argparse
import logging
import os

import mlflow
import ray
import torch
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from rmaml.datasets.episode_sampler import EpisodeSampler
from rmaml.datasets.miniimagenet import load_miniimagenet
from rmaml.datasets.synthetic import make_synthetic_dataset
from rmaml.meta_learner import RMAMLTrainer
from rmaml.models.rmaml_model import RMAMLModel
from rmaml.utils.tracking import setup_tracking

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Epochs per trial — shorter than full training
# ASHA kills bad trials at grace_period so most never reach this
TRIAL_EPOCHS = 10000
EVAL_EVERY = 500  # evaluate accuracy every N epochs inside trial


def trial_fn(trial_cfg: dict) -> None:
    """
    One HPO trial — runs inside a Ray worker.
    trial_cfg contains sampled hyperparams merged with base config.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and trainer with sampled hyperparams
    model = RMAMLModel(n_way=trial_cfg["n_way"])
    trainer = RMAMLTrainer(
        model=model,
        n_inner=trial_cfg["n_inner"],
        alpha=trial_cfg["alpha"],
        outer_lr=trial_cfg["outer_lr"],
        meta_batch=trial_cfg["meta_batch"],
        device=device,
        use_cadam=(trial_cfg["optimizer"] == "cadam"),
        use_triton=trial_cfg.get("use_triton", False),
    )

    # Dataset — use real data if available, synthetic as fallback
    data_root = trial_cfg.get("data_root", None)
    if data_root and os.path.exists(data_root):
        dataset_map = load_miniimagenet(root=data_root, split="train", augment=True)
        val_map = load_miniimagenet(root=data_root, split="val", augment=False)
    else:
        log.warning("Real data not found, using synthetic")
        dataset_map = make_synthetic_dataset(n_classes=64, images_per_class=20)
        val_map = make_synthetic_dataset(n_classes=16, images_per_class=20)

    train_sampler = EpisodeSampler(
        dataset_map,
        n_way=trial_cfg["n_way"],
        k_shot=trial_cfg["k_shot"],
        q_query=trial_cfg["q_query"],
    )
    val_sampler = EpisodeSampler(
        val_map,
        n_way=trial_cfg["n_way"],
        k_shot=trial_cfg["k_shot"],
        q_query=trial_cfg["q_query"],
    )

    # Training loop
    for epoch in range(TRIAL_EPOCHS):
        episodes = train_sampler.sample_batch(trial_cfg["meta_batch"])
        metrics = trainer.train_step(episodes)

        # Evaluate on val set periodically
        if epoch % EVAL_EVERY == 0:
            val_eps = val_sampler.sample_batch(32)
            val_metrics = trainer.evaluate(val_eps)
            orth_err = model.orthogonality_error()

            # Report to Ray Tune — ASHA uses "val_accuracy" to decide kills
            tune.report(
                {
                    "val_accuracy": val_metrics["accuracy"],
                    "meta_loss": metrics["meta_loss"],
                    "orth_error": orth_err,
                    "epoch": epoch,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/conv4_miniimagenet.yaml")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(args.config))

    ray.init(num_gpus=2)

    # Search space — sampled independently per trial
    search_space = {
        # Fixed params from base config
        "n_way": base_cfg["model"]["n_way"],
        "meta_batch": base_cfg["meta"]["meta_batch"],
        "k_shot": base_cfg["data"]["k_shot"],
        "q_query": base_cfg["data"]["q_query"],
        "data_root": os.path.abspath(base_cfg["data"]["root"]),
        # Searched params
        "alpha": tune.loguniform(0.01, 0.05),
        "outer_lr": tune.loguniform(1e-4, 5e-4),
        "n_inner": tune.choice([3, 5, 7]),
        "optimizer": "cadam",
        "use_triton": True,
    }

    if args.smoke_test:
        # Tiny smoke test — 2 trials, 100 epochs each
        search_space["data_root"] = None  # force synthetic
        n_trials = 2
        trial_epochs = 100
    else:
        n_trials = args.n_trials
        trial_epochs = TRIAL_EPOCHS

    # ASHA: kills bottom 50% at each bracket
    # grace_period = minimum epochs before a trial can be killed
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=trial_epochs,
        grace_period=1000,
        reduction_factor=2,
    )

    # Optuna for smarter search than pure random
    search_alg = OptunaSearch(
        metric="val_accuracy",
        mode="max",
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trial_fn,
            resources={"gpu": 1, "cpu": 2},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=n_trials,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=tune.RunConfig(
            name="rmaml_hpo",
            storage_path=os.path.abspath("ray_results"),
        ),
    )

    log.info(f"Starting HPO: {n_trials} trials, {trial_epochs} epochs each")
    results = tuner.fit()

    # Best trial
    best = results.get_best_result(metric="val_accuracy", mode="max")
    best_cfg = best.config
    best_acc = best.metrics["val_accuracy"]

    log.info("=" * 50)
    log.info(f"Best val accuracy: {best_acc*100:.2f}%")
    log.info("Best config:")
    log.info(f"  alpha:     {best_cfg['alpha']:.4f}")
    log.info(f"  outer_lr:  {best_cfg['outer_lr']:.6f}")
    log.info(f"  n_inner:   {best_cfg['n_inner']}")
    log.info(f"  optimizer: {best_cfg['optimizer']}")
    log.info("=" * 50)

    # Log best result to MLflow
    setup_tracking(base_cfg["experiment"])
    with mlflow.start_run(run_name="hpo_best"):
        mlflow.log_params(
            {
                "alpha": best_cfg["alpha"],
                "outer_lr": best_cfg["outer_lr"],
                "n_inner": best_cfg["n_inner"],
                "optimizer": best_cfg["optimizer"],
                "n_trials": n_trials,
            }
        )
        mlflow.log_metric("best_val_accuracy", best_acc)

    # Save best config as yaml for reuse
    best_out = {**base_cfg}
    best_out["meta"]["alpha"] = float(best_cfg["alpha"])
    best_out["meta"]["n_inner"] = int(best_cfg["n_inner"])
    best_out["optimizer"]["outer_lr"] = float(best_cfg["outer_lr"])
    best_out["optimizer"]["type"] = best_cfg["optimizer"]
    best_out["experiment"] = "rmaml_hpo_best"

    import yaml as _yaml

    with open("configs/conv4_hpo_best.yaml", "w") as f:
        _yaml.dump(best_out, f)
    log.info("Best config saved to configs/conv4_hpo_best.yaml")

    ray.shutdown()


if __name__ == "__main__":
    main()
