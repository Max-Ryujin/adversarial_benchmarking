from __future__ import annotations

from typing import Any

import torch

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
