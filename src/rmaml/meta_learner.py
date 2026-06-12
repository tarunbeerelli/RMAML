"""
RMAML bi-level optimization loop.

Inner loop: Riemannian gradient descent on support set
            (Stiefel params via retraction, Euclidean params via SGD)
Outer loop: RiemannianAdam (cAdam) or RiemannianSGD (cSGDM)
            on query set loss
"""

import geoopt
import torch
import torch.nn.functional as F
from rmaml.datasets.episode_sampler import Episode
from rmaml.kernels.stiefel_retraction import TRITON_AVAILABLE, cayley_retract
from rmaml.models.rmaml_model import RMAMLModel

_stiefel = geoopt.Stiefel()


def inner_step(
    params: dict,
    model: RMAMLModel,
    episode: Episode,
    alpha: float,
    device: torch.device,
    use_triton=False,
) -> dict:
    """
    One step of Riemannian gradient descent on the support set.

    Stiefel params  → project gradient to tangent space → retract
    Euclidean params → standard gradient step

    Args:
        params:  current parameter dict (cloned from model)
        model:   RMAMLModel (used only for functional_forward)
        episode: current task episode
        alpha:   inner loop learning rate
        device:  torch device

    Returns:
        updated parameter dict
    """
    sx = episode.support_x.to(device)
    sy = episode.support_y.to(device)

    logits = model.functional_forward(sx, params)
    loss = F.cross_entropy(logits, sy)

    # create_graph=True keeps second-order gradients for outer loop
    grads = torch.autograd.grad(
        loss,
        list(params.values()),
        create_graph=True,
        allow_unused=True,
    )

    new_params = {}
    for (name, p), g in zip(params.items(), grads):
        if g is None:
            new_params[name] = p
            continue
        if name == "classifier.weight":
            # Riemannian update: project grad → retract back to manifold
            rgrad = _stiefel.egrad2rgrad(p, g)
            if use_triton and TRITON_AVAILABLE:
                new_params[name] = cayley_retract(p, -alpha * rgrad)
            else:
                new_params[name] = _stiefel.retr(p, -alpha * rgrad)
        else:
            # Standard Euclidean gradient step
            new_params[name] = p - alpha * g

    return new_params


class RMAMLTrainer:
    """
    Manages the full RMAML training loop.

    Args:
        model:      RMAMLModel instance
        n_inner:    number of inner loop steps (paper uses 5)
        alpha:      inner loop LR (paper uses 0.1 for Conv4)
        outer_lr:   outer loop LR (paper uses 1e-3)
        meta_batch: tasks per outer update (paper uses 4)
        device:     torch device
        use_cadam:  True = cAdam (default), False = cSGDM (ablation)
    """

    def __init__(
        self,
        model: RMAMLModel,
        n_inner: int = 5,
        alpha: float = 0.1,
        outer_lr: float = 1e-3,
        meta_batch: int = 4,
        device: torch.device = torch.device("cpu"),
        use_cadam: bool = True,
        use_triton=False,
    ):
        self.model = model.to(device)
        self.n_inner = n_inner
        self.alpha = alpha
        self.meta_batch = meta_batch
        self.device = device
        self.use_triton = use_triton

        # Outer optimizer — geoopt handles Stiefel + Euclidean params
        # RiemannianAdam = cAdam from paper
        # RiemannianSGD  = cSGDM from paper (ablation study)
        if use_cadam:
            self.outer_opt = geoopt.optim.RiemannianAdam(
                self.model.parameters(), lr=outer_lr
            )
        else:
            self.outer_opt = geoopt.optim.RiemannianSGD(
                self.model.parameters(), lr=outer_lr, momentum=0.9
            )

    def _task_loss(self, episode: Episode) -> torch.Tensor:
        """
        Run inner loop on support set, return query loss.
        This is the core of RMAML — Algorithm 1 from the paper.
        """
        # Clone params — each task gets its own adapted copy
        params = {n: p.clone() for n, p in self.model.named_parameters()}

        # Inner loop: k Riemannian gradient steps on support set
        for _ in range(self.n_inner):
            params = inner_step(params, self.model, episode, self.alpha, self.device)

        # Evaluate adapted params on query set
        qx = episode.query_x.to(self.device)
        qy = episode.query_y.to(self.device)
        logits = self.model.functional_forward(qx, params)
        return F.cross_entropy(logits, qy)

    def train_step(self, episodes: list[Episode]) -> dict:
        """
        One outer-loop meta-update over a batch of tasks.
        Returns dict of metrics for logging.
        """
        self.model.train()
        self.outer_opt.zero_grad()

        # Average query loss across all tasks in the meta-batch
        meta_loss = sum(self._task_loss(ep) for ep in episodes)
        meta_loss = meta_loss / len(episodes)

        meta_loss.backward()
        self.outer_opt.step()

        return {"meta_loss": meta_loss.item()}

    @torch.no_grad()
    def evaluate(self, episodes: list[Episode]) -> dict:
        """
        Meta-test accuracy: adapt to support set, score on query set.
        Uses no_grad for speed — inner loop still runs but no graph kept.
        """
        self.model.eval()
        accs = []

        for ep in episodes:
            # Re-enable grad just for inner loop adaptation
            with torch.enable_grad():
                params = {n: p.clone() for n, p in self.model.named_parameters()}
                for _ in range(self.n_inner):
                    params = inner_step(params, self.model, ep, self.alpha, self.device)

            qx = ep.query_x.to(self.device)
            qy = ep.query_y.to(self.device)
            logits = self.model.functional_forward(qx, params)
            acc = (logits.argmax(1) == qy).float().mean().item()
            accs.append(acc)

        return {"accuracy": sum(accs) / len(accs)}
