from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor

from adversarial_benchmarking.data.image_folder import load_image_tensor
from adversarial_benchmarking.logging_utils import configure_logging, get_logger
from adversarial_benchmarking.models.qwen3_vl import Qwen3VLFirstTokenClassifier
from adversarial_benchmarking.tasks.first_token import FirstTokenTask


logger = get_logger("scripts.prefill_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Context editor: inject a forced assistant prefix (prefill) and see how the "
        "model continues, WITHOUT any attack. Use it to find which (question, opening) pairs "
        "actually jailbreak before spending compute on the APGD attack.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="The (potentially refused) question.")
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Forced beginning of the assistant's reply, e.g. 'Sure, here is how'. Repeat the "
        "flag to try several prefixes against the same question. An empty string ('') probes the "
        "unmodified model.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct", help="HF model name.")
    parser.add_argument("--device", default="auto", help="Torch device ('auto' picks an accelerator or CPU).")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Model weight dtype. 'auto' uses float32 on CPU and the checkpoint dtype on GPU.",
    )
    parser.add_argument("--resize", type=int, default=448, help="Fixed square resize before the processor.")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Tokens generated after the prefix.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(debug=args.debug)

    prefixes = args.prefix if args.prefix is not None else [""]

    processor = AutoProcessor.from_pretrained(args.model_name)
    # The task only supplies the prompt text here; the target token is unused for generation,
    # so any non-empty placeholder is fine.
    task = FirstTokenTask.from_target(
        instruction=args.prompt,
        target_text="Sure",
        tokenizer=processor.tokenizer,
    )

    model = Qwen3VLFirstTokenClassifier(
        model_name=args.model_name,
        task=task,
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        resize=args.resize,
        torch_dtype=args.dtype,
    )

    image = load_image_tensor(Path(args.image)).unsqueeze(0).to(model.device)

    print(f"Question: {args.prompt}\n")
    for prefix in prefixes:
        model.set_assistant_prefix(prefix)
        continuation = model.generate_text(image, max_new_tokens=args.max_new_tokens)[0]
        shown_prefix = prefix if prefix else "(none)"
        print("=" * 72)
        print(f"Forced prefix: {shown_prefix!r}")
        print("-" * 72)
        print(f"{prefix}{continuation}")
    print("=" * 72)


if __name__ == "__main__":
    main()
