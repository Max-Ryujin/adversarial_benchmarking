from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch


LOGGER_NAME = "adversarial_benchmarking"


def get_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(debug: bool = False, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging configured")
    return logger


def summarize_tensor(tensor: torch.Tensor) -> str:
    detached = tensor.detach()
    stats_tensor = detached.float()
    return (
        f"shape={tuple(detached.shape)} dtype={detached.dtype} device={detached.device} "
        f"min={stats_tensor.min().item():.6f} max={stats_tensor.max().item():.6f} "
        f"mean={stats_tensor.mean().item():.6f} std={stats_tensor.std(unbiased=False).item():.6f}"
    )


def format_named_logits(names: list[str], logits: torch.Tensor) -> str:
    values = logits.detach().cpu().tolist()
    return ", ".join(f"{name}={value:.4f}" for name, value in zip(names, values))


def format_topk_vocab_logits(
    vocab_logits: torch.Tensor,
    tokenizer: Any,
    k: int = 10,
) -> str:
    top_k = min(k, vocab_logits.shape[-1])
    top_values, top_indices = torch.topk(vocab_logits.detach(), k=top_k, dim=-1)
    entries: list[str] = []
    for token_id, value in zip(top_indices[0].tolist(), top_values[0].tolist()):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_text = token_text.replace("\n", "\\n")
        entries.append(f"id={token_id} token={token_text!r} logit={value:.4f}")
    return "; ".join(entries)