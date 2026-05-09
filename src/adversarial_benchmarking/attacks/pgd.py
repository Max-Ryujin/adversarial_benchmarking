from __future__ import annotations

import torch
import torch.nn.functional as F

from adversarial_benchmarking.config import AttackConfig


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: AttackConfig,
) -> torch.Tensor:
    if config.random_start:
        delta = torch.empty_like(images).uniform_(-config.epsilon, config.epsilon)
        adv_images = torch.clamp(images + delta, 0.0, 1.0)
    else:
        adv_images = images.clone()

    for _ in range(config.steps):
        adv_images.requires_grad_(True)
        logits = model(adv_images)
        loss = F.cross_entropy(logits, labels)
        if config.targeted:
            loss = -loss

        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + config.step_size * grad.sign()
        delta = torch.clamp(adv_images - images, min=-config.epsilon, max=config.epsilon)
        adv_images = torch.clamp(images + delta, 0.0, 1.0)

    return adv_images.detach()
