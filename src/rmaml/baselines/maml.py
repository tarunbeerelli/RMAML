"""
Vanilla MAML baseline — Euclidean version of RMAML.
Used to reproduce Table 3 from the paper and show RMAML's improvement.

Key difference from RMAML:
- No Stiefel constraint — classifier weight is a plain nn.Parameter
- Inner loop: standard SGD (no retraction, no projection)
- Outer loop: standard Adam or SGD (no Riemannian ops)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rmaml.datasets.episode_sampler import Episode
from rmaml.models.backbone import Conv4
from torch import Tensor


class MAMLModel(nn.Module):
    """
    Conv4 + standard FC head. Identical to RMAMLModel but
    classifier weight is Euclidean (plain nn.Parameter).
    """

    def __init__(self, n_way: int = 5):
        super().__init__()
        self.encoder = Conv4()
        # Standard linear layer — no Stiefel constraint
        self.classifier = nn.Linear(self.encoder.out_dim, n_way)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))

    def functional_forward(self, x: Tensor, params: dict) -> Tensor:
        h = self.encoder.functional_forward(x, params)
        return F.linear(
            h,
            params["classifier.weight"],
            params["classifier.bias"],
        )


def maml_inner_step(
    params: dict,
    model: MAMLModel,
    episode: Episode,
    alpha: float,
    device: torch.device,
) -> dict:
    """
    Standard Euclidean inner step — no Riemannian ops at all.
    All params updated with plain gradient descent.
    """
    sx = episode.support_x.to(device)
    sy = episode.support_y.to(device)

    logits = model.functional_forward(sx, params)
    loss = F.cross_entropy(logits, sy)

    grads = torch.autograd.grad(
        loss,
        list(params.values()),
        create_graph=True,
        allow_unused=True,
    )

    return {
        name: p - alpha * g if g is not None else p
        for (name, p), g in zip(params.items(), grads)
    }


class MAMLTrainer:
    """
    Vanilla MAML trainer — direct Euclidean counterpart to RMAMLTrainer.

    Args:
        model:      MAMLModel instance
        n_inner:    inner loop steps
        alpha:      inner loop LR
        outer_lr:   outer loop LR
        meta_batch: tasks per outer update
        device:     torch device
        optimizer:  "adam" | "sgd"
    """

    def __init__(
        self,
        model: MAMLModel,
        n_inner: int = 5,
        alpha: float = 0.1,
        outer_lr: float = 1e-3,
        meta_batch: int = 4,
        device: torch.device = torch.device("cpu"),
        optimizer: str = "adam",
    ):
        self.model = model.to(device)
        self.n_inner = n_inner
        self.alpha = alpha
        self.meta_batch = meta_batch
        self.device = device

        # Standard Euclidean optimizers
        if optimizer == "adam":
            self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        else:
            self.outer_opt = torch.optim.SGD(
                self.model.parameters(), lr=outer_lr, momentum=0.9
            )

    def _task_loss(self, episode: Episode) -> torch.Tensor:
        params = {n: p.clone() for n, p in self.model.named_parameters()}
        for _ in range(self.n_inner):
            params = maml_inner_step(
                params, self.model, episode, self.alpha, self.device
            )
        qx = episode.query_x.to(self.device)
        qy = episode.query_y.to(self.device)
        return F.cross_entropy(self.model.functional_forward(qx, params), qy)

    def train_step(self, episodes: list[Episode]) -> dict:
        self.model.train()
        self.outer_opt.zero_grad()
        meta_loss = sum(self._task_loss(ep) for ep in episodes) / len(episodes)
        meta_loss.backward()
        self.outer_opt.step()
        return {"meta_loss": meta_loss.item()}

    @torch.no_grad()
    def evaluate(self, episodes: list[Episode]) -> dict:
        self.model.eval()
        accs = []
        for ep in episodes:
            with torch.enable_grad():
                params = {n: p.clone() for n, p in self.model.named_parameters()}
                for _ in range(self.n_inner):
                    params = maml_inner_step(
                        params, self.model, ep, self.alpha, self.device
                    )
            qx = ep.query_x.to(self.device)
            qy = ep.query_y.to(self.device)
            logits = self.model.functional_forward(qx, params)
            accs.append((logits.argmax(1) == qy).float().mean().item())
        return {"accuracy": sum(accs) / len(accs)}
