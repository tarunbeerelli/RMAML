"""
MLflow tracking wrapper.
Single import point — swap to W&B later by changing only this file.
"""

import os

import mlflow


def setup_tracking(experiment_name: str) -> None:
    """Call once at the top of train.py."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


class RunTracker:
    """
    Context manager wrapping one MLflow run.

    Usage:
        with RunTracker(cfg) as tracker:
            tracker.log(step=0, meta_loss=0.5, orth_error=1e-6)
    """

    def __init__(self, cfg: dict, run_name: str | None = None):
        self.cfg = cfg
        self.run_name = run_name

    def __enter__(self):
        self._run = mlflow.start_run(run_name=self.run_name)
        mlflow.log_params(self._flatten(self.cfg))
        return self

    def __exit__(self, *args):
        mlflow.end_run()

    def log(self, step: int, **metrics: float) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    @staticmethod
    def _flatten(cfg: dict, prefix: str = "") -> dict:
        """Flatten nested config dict for MLflow param logging."""
        out = {}
        for k, v in cfg.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(RunTracker._flatten(v, key))
            else:
                out[key] = v
        return out
