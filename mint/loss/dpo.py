import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def compute_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_mask = labels != -100

    labels_stable = labels.clone()
    labels_stable[labels_stable == -100] = 0

    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=labels_stable.unsqueeze(-1)).squeeze(-1)

    return (per_token_logps * loss_mask).sum(dim=-1)


class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(  # noqa: ANN201
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)

        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
            reward_margins = chosen_rewards - rejected_rewards
            accuracy = (pi_logratios > ref_logratios).float().mean()

        return losses.mean(), {
            "loss": losses.mean().item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margins": reward_margins.mean().item(),
            "accuracy": accuracy.item(),
        }
