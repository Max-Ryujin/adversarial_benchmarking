from __future__ import annotations

from typing import Any

import torch


def build_autoattack(model: torch.nn.Module, epsilon: float, norm: str = "Linf", **kwargs: Any) -> Any:
    try:
        from autoattack import AutoAttack
    except ImportError as exc:
        raise RuntimeError(
            "AutoAttack is not installed. Install the optional dependency with `pip install .[autoattack]`."
        ) from exc

    return AutoAttack(model, norm=norm, eps=epsilon, **kwargs)
