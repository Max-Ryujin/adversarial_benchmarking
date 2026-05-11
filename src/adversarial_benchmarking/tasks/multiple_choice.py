from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from transformers import PreTrainedTokenizerBase

from adversarial_benchmarking.logging_utils import format_named_logits, get_logger
from adversarial_benchmarking.tasks.base import ChoiceOption, TaskSpec


LETTER_CHOICES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


logger = get_logger("tasks.multiple_choice")


@dataclass(slots=True)
class MultipleChoiceTask(TaskSpec):
    instruction: str
    labels: list[str]
    options: list[ChoiceOption]

    @classmethod
    def from_labels(
        cls,
        instruction: str,
        labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> "MultipleChoiceTask":
        if len(labels) > len(LETTER_CHOICES):
            raise ValueError("Too many labels for single-letter multiple choice.")

        options: list[ChoiceOption] = []
        for idx, label in enumerate(labels):
            letter = LETTER_CHOICES[idx]
            token_ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Choice letter {letter!r} is not a single token: {token_ids}")
            options.append(ChoiceOption(letter=letter, label=label, token_id=token_ids[0]))

        logger.info("Built multiple-choice task with %s options", len(options))
        logger.debug(
            "Task options: %s",
            ", ".join(f"{option.letter}:{option.label}:{option.token_id}" for option in options),
        )
        return cls(instruction=instruction, labels=labels, options=options)

    def build_prompt(self) -> str:
        option_lines = [f"{option.letter}: {option.label}" for option in self.options]
        options_text = "\n".join(option_lines)
        prompt = (
            f"{self.instruction}\n"
            f"Choose exactly one option and answer with only the letter.\n"
            f"{options_text}"
        )
        logger.debug("Built prompt: %s", prompt)
        return prompt

    def class_labels(self) -> list[str]:
        return [option.label for option in self.options]

    def class_token_ids(self) -> list[int]:
        return [option.token_id for option in self.options if option.token_id is not None]

    def class_index_for_label(self, label: str) -> int:
        for index, option in enumerate(self.options):
            if option.label == label or option.letter == label:
                logger.debug("Resolved label %r to index %s (%s)", label, index, option.letter)
                return index
        raise KeyError(f"Unknown class label: {label}")

    def class_logits_from_vocab(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        token_ids = torch.tensor(self.class_token_ids(), device=vocab_logits.device)
        class_logits = vocab_logits.index_select(dim=-1, index=token_ids)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Selected class logits from vocab: %s",
                format_named_logits(self.class_labels(), class_logits[0]),
            )
        return class_logits
