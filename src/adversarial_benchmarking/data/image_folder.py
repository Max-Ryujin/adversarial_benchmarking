from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from adversarial_benchmarking.logging_utils import get_logger, summarize_tensor


logger = get_logger("data.image_folder")


def load_image_tensor(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    logger.info("Loaded image from %s", image_path)
    logger.debug("Image size=%s tensor=%s", image.size, summarize_tensor(tensor))
    return tensor
