from __future__ import annotations

import logging
from typing import Any

import torch

from adversarial_benchmarking.config import AttackConfig
from adversarial_benchmarking.logging_utils import get_logger


logger = get_logger("attacks.autoattack")


def build_autoattack(model: torch.nn.Module, epsilon: float, norm: str = "Linf", **kwargs: Any) -> Any:
    try:
        from autoattack import AutoAttack
    except ImportError as exc:
        raise RuntimeError(
            "AutoAttack is not installed. Install the optional dependency with `pip install .[autoattack]`."
        ) from exc

    logger.info("Building AutoAttack with norm=%s epsilon=%s", norm, epsilon)
    logger.debug("AutoAttack kwargs: %s", kwargs)
    return AutoAttack(model, norm=norm, eps=epsilon, **kwargs)


def autoattack_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: AttackConfig,
) -> torch.Tensor:
    if config.targeted:
        raise ValueError("AutoAttack support is currently limited to untargeted evaluation.")

    logger.info(
        "Starting AutoAttack: norm=%s epsilon=%.6f targeted=%s",
        config.norm,
        config.epsilon,
        config.targeted,
    )
    adversary = build_autoattack(
        model,
        epsilon=config.epsilon,
        norm=config.norm,
        version="standard",
        device=str(images.device),
        verbose=logger.isEnabledFor(logging.DEBUG),
    )
    adv_images = adversary.run_standard_evaluation(images, labels, bs=images.shape[0])
    logger.info("Finished AutoAttack")
    return adv_images.detach()
