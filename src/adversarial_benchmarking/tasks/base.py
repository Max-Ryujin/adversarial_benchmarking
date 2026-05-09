from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    letter: str
    label: str
    token_id: int | None = None


class TaskSpec:
    def build_prompt(self) -> str:
        raise NotImplementedError

    def class_labels(self) -> list[str]:
        raise NotImplementedError

    def class_token_ids(self) -> list[int]:
        raise NotImplementedError

    def class_index_for_label(self, label: str) -> int:
        raise NotImplementedError

    def class_logits_from_vocab(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
