from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from adversarial_benchmarking.config import AttackConfig
from adversarial_benchmarking.logging_utils import format_named_logits, get_logger, summarize_tensor


logger = get_logger("attacks.pgd")


def pgd_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: AttackConfig,
) -> torch.Tensor:
    logger.info(
        "Starting PGD attack: steps=%s epsilon=%.6f step_size=%.6f targeted=%s random_start=%s",
        config.steps,
        config.epsilon,
        config.step_size,
        config.targeted,
        config.random_start,
    )
    debug_enabled = logger.isEnabledFor(logging.DEBUG)

    if config.random_start:
        delta = torch.empty_like(images).uniform_(-config.epsilon, config.epsilon)
        adv_images = torch.clamp(images + delta, 0.0, 1.0)
        if debug_enabled:
            logger.debug("Initialized random start delta: %s", summarize_tensor(delta))
    else:
        adv_images = images.clone()
        logger.debug("Using deterministic PGD start")

    task = getattr(model, "task", None)

    for step_index in range(config.steps):
        adv_images.requires_grad_(True)
        logits = model(adv_images)
        loss = F.cross_entropy(logits, labels)
        if config.targeted:
            loss = -loss

        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + config.step_size * grad.sign()
        delta = torch.clamp(adv_images - images, min=-config.epsilon, max=config.epsilon) # ensure epsilon ball
        adv_images = torch.clamp(images + delta, 0.0, 1.0)
        # The per-step diagnostics below force host<->device syncs (`.item()`, `.cpu()`)
        # and extra reductions, so only compute them when DEBUG logging is on.
        if debug_enabled:
            predicted_indices = logits.argmax(dim=-1)
            logger.debug(
                "PGD step %s/%s loss=%.6f predicted_indices=%s grad=%s delta=%s",
                step_index + 1,
                config.steps,
                loss.item(),
                predicted_indices.detach().cpu().tolist(),
                summarize_tensor(grad),
                summarize_tensor(delta),
            )
            if task is not None:
                logger.debug(
                    "PGD step %s class logits: %s",
                    step_index + 1,
                    format_named_logits(task.class_labels(), logits[0]),
                )

    logger.info("Finished PGD attack")
    return adv_images.detach()
