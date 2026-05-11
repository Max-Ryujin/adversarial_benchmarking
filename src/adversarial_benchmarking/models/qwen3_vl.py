from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from adversarial_benchmarking.logging_utils import (
    format_named_logits,
    format_topk_vocab_logits,
    get_logger,
    summarize_tensor,
)
from adversarial_benchmarking.tasks.base import TaskSpec


@dataclass(slots=True)
class ForwardResult:
    class_logits: torch.Tensor
    vocab_logits: torch.Tensor
    inputs: dict[str, Any]


logger = get_logger("models.qwen3_vl")


def _resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available in this PyTorch build; falling back to CPU.")
        return torch.device("cpu")
    if resolved.type == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            logger.warning("MPS was requested but is not available; falling back to CPU.")
            return torch.device("cpu")
    return resolved


class Qwen3VLFirstTokenClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        task: TaskSpec,
        device: str = "auto",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        resize: int | None = 448,
        torch_dtype: torch.dtype | str = "auto",
    ) -> None:
        super().__init__()
        processor_kwargs: dict[str, Any] = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        resolved_device = _resolve_device(device)
        self.model.to(resolved_device)
        self.model.eval()

        self.device_name = str(resolved_device)
        self.task = task
        self.resize = resize
        self.prompt_text = self._build_chat_prompt(task.build_prompt())
        logger.info("Initialized Qwen3-VL classifier for %s on %s", model_name, self.device)
        logger.debug(
            "Model config: resize=%s min_pixels=%s max_pixels=%s classes=%s",
            resize,
            min_pixels,
            max_pixels,
            task.class_labels(),
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _build_chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _prepare_images(self, images: torch.Tensor) -> list[torch.Tensor]:
        if images.dim() != 4:
            raise ValueError(f"Expected BCHW image tensor, got shape {tuple(images.shape)}")

        image_batch = images.clamp(0.0, 1.0)
        logger.debug("Preparing image batch: %s", summarize_tensor(image_batch))
        if self.resize is not None:
            image_batch = F.interpolate(
                image_batch,
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            )
            logger.debug("Resized image batch to %sx%s: %s", self.resize, self.resize, summarize_tensor(image_batch))

        return [image_tensor for image_tensor in image_batch]

    def _processor_inputs(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        prepared_images = self._prepare_images(images)
        texts = [self.prompt_text] * len(prepared_images)
        inputs = self.processor(
            images=prepared_images,
            text=texts,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
        )
        inputs.pop("token_type_ids", None)
        moved_inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        logger.debug(
            "Processor inputs prepared: %s",
            ", ".join(
                f"{key}={tuple(value.shape) if hasattr(value, 'shape') else type(value).__name__}"
                for key, value in moved_inputs.items()
            ),
        )
        return moved_inputs

    def forward_result(self, images: torch.Tensor) -> ForwardResult:
        inputs = self._processor_inputs(images)
        outputs = self.model(**inputs, logits_to_keep=1)
        vocab_logits = outputs.logits[:, -1, :]
        class_logits = self.task.class_logits_from_vocab(vocab_logits)
        logger.debug("Forward class logits: %s", format_named_logits(self.task.class_labels(), class_logits[0]))
        logger.debug(
            "Top vocab logits for first answer token: %s",
            format_topk_vocab_logits(vocab_logits[:1], self.processor.tokenizer),
        )
        return ForwardResult(class_logits=class_logits, vocab_logits=vocab_logits, inputs=inputs)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_result(images).class_logits

    @torch.inference_mode()
    def generate_letters(self, images: torch.Tensor, max_new_tokens: int = 4) -> list[str]:
        inputs = self._processor_inputs(images)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
        trimmed = [output_ids[prompt_length:] for output_ids, prompt_length in zip(generated_ids, prompt_lengths)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logger.debug("Generated token ids: %s", [token_ids.tolist() for token_ids in trimmed])
        logger.debug("Decoded generations: %s", decoded)
        return decoded
