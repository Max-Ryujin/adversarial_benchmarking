from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image_tensor(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return tensor
