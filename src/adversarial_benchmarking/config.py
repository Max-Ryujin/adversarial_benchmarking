from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AttackConfig:
    epsilon: float = 8.0 / 255.0
    step_size: float = 2.0 / 255.0
    steps: int = 20
    targeted: bool = False
    random_start: bool = True


@dataclass(slots=True)
class RunConfig:
    image_path: Path
    output_dir: Path
    true_label: str
    prompt: str
    choices: list[str]
    debug: bool = False
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    device: str = "auto"
    min_pixels: int | None = None
    max_pixels: int | None = None
    resize: int | None = 448
    target_label: str | None = None
