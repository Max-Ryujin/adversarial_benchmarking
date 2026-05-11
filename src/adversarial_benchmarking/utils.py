from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from adversarial_benchmarking.logging_utils import get_logger, summarize_tensor


logger = get_logger("utils")


def save_image_tensor(image: torch.Tensor, path: str | Path) -> None:
    image_uint8 = (image.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
    pil_image = Image.fromarray(image_uint8.permute(1, 2, 0).numpy(), mode="RGB")
    pil_image.save(path)
    logger.info("Saved image artifact to %s", path)
    logger.debug("Saved image stats: %s", summarize_tensor(image))


def write_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote JSON artifact to %s", path)
    logger.debug("JSON keys: %s", sorted(data.keys()))
