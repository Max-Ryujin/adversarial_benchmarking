from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.type == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            return torch.device("cpu")
    return resolved


def resolve_torch_dtype(
    torch_dtype: torch.dtype | str,
    device: torch.device,
) -> torch.dtype | str:
    if device.type == "cpu" and torch_dtype == "auto":
        return torch.float32
    return torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Qwen3-VL about a single image.")
    parser.add_argument("image", help="Path to the image shown to the model.")
    parser.add_argument("--prompt", help="Optional first user prompt.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct", help="HF model name.")
    parser.add_argument("--device", default="auto", help="Torch device. Use 'auto' to pick a supported accelerator or CPU.")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens generated for each assistant reply.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    return parser.parse_args()


def build_user_message(text: str, include_image: bool) -> dict[str, object]:
    content: list[dict[str, str]] = []
    if include_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def render_prompt(processor: AutoProcessor, messages: list[dict[str, object]]) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_reply(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    messages: list[dict[str, object]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt_text = render_prompt(processor, messages)
    inputs = processor(
        images=[image],
        text=[prompt_text],
        return_tensors="pt",
        padding=True,
    )
    inputs.pop("token_type_ids", None)
    model_inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = int(model_inputs["attention_mask"][0].sum().item())
    reply_ids = generated_ids[0][prompt_length:]
    return processor.decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def prompt_for_input(prompt_text: str) -> str | None:
    try:
        user_input = input(prompt_text).strip()
    except EOFError:
        return None
    if user_input.lower() in {"/exit", "/quit", "exit", "quit"}:
        return None
    return user_input


def main() -> None:
    args = parse_args()

    resolved_device = resolve_device(args.device)
    resolved_dtype = resolve_torch_dtype("auto", resolved_device)
    processor_kwargs: dict[str, int] = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=resolved_dtype,
        device_map=None,
    )
    model.to(resolved_device)
    model.eval()

    image = Image.open(Path(args.image)).convert("RGB")
    messages: list[dict[str, object]] = []
    include_image = True

    print(f"Loaded {args.image} on {model.device}. Type /exit to stop.")

    first_prompt = args.prompt
    while True:
        if first_prompt is not None:
            user_text = first_prompt.strip()
            first_prompt = None
            if not user_text:
                continue
            print(f"You: {user_text}")
        else:
            prompted = prompt_for_input("You: ")
            if prompted is None:
                break
            if not prompted:
                continue
            user_text = prompted

        messages.append(build_user_message(user_text, include_image=include_image))
        include_image = False
        reply = generate_reply(
            model=model,
            processor=processor,
            image=image,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()