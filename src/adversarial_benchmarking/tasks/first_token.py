from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from adversarial_benchmarking.logging_utils import get_logger
from adversarial_benchmarking.tasks.base import TaskSpec


logger = get_logger("tasks.first_token")


@dataclass(slots=True)
class FirstTokenTask(TaskSpec):
    """Free-generation task whose objective is the *first generated token*.

    Unlike ``MultipleChoiceTask`` this does not restrict the answer to a handful of letters.
    ``class_logits_from_vocab`` returns the **full vocabulary logits**, so a targeted
    cross-entropy toward ``target_token_id`` is exactly "maximize the probability that the
    first answer token is the target" — the objective a jailbreak wants (push the model into an
    affirmative opening like ``Sure`` / ``Yes`` and let it continue from there).

    It deliberately does not slice to a small candidate set: a single-class softmax is
    degenerate (loss is always 0, gradient is 0), so the contrast against the rest of the
    vocabulary is what produces a usable gradient.
    """

    instruction: str
    target_text: str
    target_token_id: int

    @classmethod
    def from_target(
        cls,
        instruction: str,
        target_text: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> "FirstTokenTask":
        token_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Target text {target_text!r} did not encode to any tokens.")
        if len(token_ids) > 1:
            decoded_first = tokenizer.decode(token_ids[:1])
            logger.warning(
                "Target %r encodes to %s tokens %s; attacking only the FIRST token %r (id %s). "
                "Forcing one token rarely guarantees the rest of the opening — pick a target whose "
                "first token already commits the model (e.g. 'Sure', 'Yes').",
                target_text,
                len(token_ids),
                token_ids,
                decoded_first,
                token_ids[0],
            )
        target_token_id = token_ids[0]
        logger.info(
            "Built first-token task targeting %r -> token id %s (%r)",
            target_text,
            target_token_id,
            tokenizer.decode([target_token_id]),
        )
        return cls(
            instruction=instruction,
            target_text=target_text,
            target_token_id=target_token_id,
        )

    def build_prompt(self) -> str:
        return self.instruction

    def class_labels(self) -> list[str]:
        # Full vocabulary is the class space; there is no small human-readable label list.
        # Callers that need to name a token id should decode it via the tokenizer instead.
        return []

    def class_token_ids(self) -> list[int]:
        return [self.target_token_id]

    def class_index_for_label(self, label: str) -> int:
        # The "class index" over the full-vocab logits is just the target token id.
        return self.target_token_id

    def class_logits_from_vocab(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        # Identity: the full vocabulary IS the class space for a first-token objective.
        return vocab_logits
