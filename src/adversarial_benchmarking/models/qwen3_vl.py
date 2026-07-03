from __future__ import annotations

import gc
import logging
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


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def free_memory(device: torch.device | None = None) -> None:
    """Release cached allocator memory. Helps both CPU RAM and (ROCm/CUDA) VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        # ROCm exposes the same `torch.cuda` namespace as CUDA.
        torch.cuda.empty_cache()


def prepare_image_batch(images: torch.Tensor, resize: int | None) -> torch.Tensor:
    """Clamp a BCHW float batch to [0, 1] and optionally bilinearly resize it to a square.

    This is the single source of truth for how raw pixels are transformed before the Qwen
    processor sees them. Both the attack (`Qwen3VLFirstTokenClassifier`) and the chat script
    call it so an image is preprocessed identically in both, leaving the prompt as the only
    difference between them.
    """
    if images.dim() != 4:
        raise ValueError(f"Expected BCHW image tensor, got shape {tuple(images.shape)}")

    image_batch = images.clamp(0.0, 1.0)
    if resize is not None:
        image_batch = F.interpolate(
            image_batch,
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
    return image_batch


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


def _resolve_torch_dtype(
    torch_dtype: torch.dtype | str,
    device: torch.device,
) -> torch.dtype | str:
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    key = str(torch_dtype).lower()
    if key in _DTYPE_ALIASES:
        resolved = _DTYPE_ALIASES[key]
        if device.type == "cpu" and resolved in (torch.float16, torch.bfloat16):
            # fp16/bf16 backward passes are unsupported or extremely slow on CPU; the
            # attacks need a backward pass through the model, so keep CPU in float32.
            logger.warning(
                "Requested %s on CPU is not usable for gradient-based attacks; using float32 instead.",
                key,
            )
            return torch.float32
        return resolved

    if key == "auto":
        if device.type == "cpu":
            logger.info("Using float32 weights on CPU to keep backward passes supported.")
            return torch.float32
        # On an accelerator, let transformers pick the checkpoint's native dtype.
        return "auto"

    raise ValueError(f"Unsupported torch dtype: {torch_dtype!r}")


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
        grad_checkpointing: bool = False,
        assistant_prefix: str = "",
    ) -> None:
        super().__init__()
        processor_kwargs: dict[str, Any] = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_torch_dtype(torch_dtype, resolved_device)

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        self.model.to(resolved_device)
        self.model.eval()

        # The attacks only differentiate w.r.t. the input image, never the weights.
        # Freezing the parameters avoids allocating parameter-gradient buffers and lets
        # autograd prune bookkeeping it would otherwise keep for trainable leaves.
        self.model.requires_grad_(False)

        self.grad_checkpointing = grad_checkpointing
        if grad_checkpointing:
            # Trades ~30% extra compute for a large drop in peak activation memory during
            # the attack backward pass. Output logits are mathematically unchanged.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.config.use_cache = False
            logger.info("Gradient checkpointing enabled (lower memory, slower per step).")

        self.device_name = str(resolved_device)
        self.task = task
        self.resize = resize
        self.assistant_prefix = assistant_prefix
        self.prompt_text = self._build_chat_prompt(task.build_prompt(), assistant_prefix)
        logger.info("Initialized Qwen3-VL classifier for %s on %s", model_name, self.device)
        logger.debug(
            "Model config: dtype=%s resize=%s min_pixels=%s max_pixels=%s grad_checkpointing=%s classes=%s",
            next(self.model.parameters()).dtype,
            resize,
            min_pixels,
            max_pixels,
            grad_checkpointing,
            task.class_labels(),
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _build_chat_prompt(self, user_prompt: str, assistant_prefix: str = "") -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # A prefill: text appended after the "<|im_start|>assistant\n" header becomes the forced
        # beginning of the reply. Generation (and the first-token attack) then continues from the
        # token right after it, so the objective is the first *free* token past the prefix.
        if assistant_prefix:
            prompt = prompt + assistant_prefix
        return prompt

    def set_assistant_prefix(self, assistant_prefix: str) -> None:
        """Rebuild the prompt with a new forced assistant prefix (cheap; no model reload)."""
        self.assistant_prefix = assistant_prefix
        self.prompt_text = self._build_chat_prompt(self.task.build_prompt(), assistant_prefix)

    def _prepare_images(self, images: torch.Tensor) -> list[torch.Tensor]:
        image_batch = prepare_image_batch(images, self.resize)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Prepared image batch: %s", summarize_tensor(image_batch))

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
        if logger.isEnabledFor(logging.DEBUG):
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
        # `logits_to_keep=1` restricts the LM head to the final prompt position, and
        # `use_cache=False` avoids building a KV cache we never read on a single forward.
        outputs = self.model(**inputs, logits_to_keep=1, use_cache=False)
        vocab_logits = outputs.logits[:, -1, :]
        class_logits = self.task.class_logits_from_vocab(vocab_logits)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Forward class logits: %s", format_named_logits(self.task.class_labels(), class_logits[0]))
            logger.debug(
                "Top vocab logits for first answer token: %s",
                format_topk_vocab_logits(vocab_logits[:1], self.processor.tokenizer),
            )
        return ForwardResult(class_logits=class_logits, vocab_logits=vocab_logits, inputs=inputs)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_result(images).class_logits

    @torch.inference_mode()
    def predict_result(self, images: torch.Tensor) -> ForwardResult:
        """Gradient-free forward for evaluation/logging (no autograd graph is built)."""
        return self.forward_result(images)

    @torch.inference_mode()
    def generate_letters(self, images: torch.Tensor, max_new_tokens: int = 4) -> list[str]:
        inputs = self._processor_inputs(images)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
        trimmed = [output_ids[prompt_length:] for output_ids, prompt_length in zip(generated_ids, prompt_lengths)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Generated token ids: %s", [token_ids.tolist() for token_ids in trimmed])
            logger.debug("Decoded generations: %s", decoded)
        return decoded

    @torch.inference_mode()
    def generate_text(self, images: torch.Tensor, max_new_tokens: int = 256) -> list[str]:
        """Greedily generate a full continuation for each image.

        The forced ``assistant_prefix`` (if any) is part of the prompt, so the returned text is
        only what the model produced *after* the prefix. Prepend ``self.assistant_prefix`` when
        showing the reply the way the model "sees" it.
        """
        inputs = self._processor_inputs(images)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        prompt_lengths = inputs["attention_mask"].sum(dim=-1).tolist()
        trimmed = [output_ids[prompt_length:] for output_ids, prompt_length in zip(generated_ids, prompt_lengths)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Generated continuation token ids: %s", [token_ids.tolist() for token_ids in trimmed])
            logger.debug("Decoded continuations: %s", decoded)
        return decoded
