from __future__ import annotations

import argparse
import time
from pathlib import Path
from threading import Thread

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextIteratorStreamer

from adversarial_benchmarking.data.image_folder import image_to_tensor
from adversarial_benchmarking.models.qwen3_vl import (
    _resolve_device,
    _resolve_torch_dtype,
    free_memory,
    prepare_image_batch,
)


HELP_TEXT = """\
Commands:
  /help                 show this help
  /exit, /quit          leave the chat
  /reset                clear the conversation (keep the same image)
  /image <path>         load a different image and start a fresh conversation
  /system <text>        set a system prompt and start a fresh conversation
  /regen                regenerate the last assistant reply
  /save <path>          write the transcript to a text file
  /info                 show the current model / image / decoding settings
Anything else is sent to the model as your next message.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Qwen3-VL about a single image.")
    parser.add_argument("image", help="Path to the image shown to the model.")
    parser.add_argument("--prompt", help="Optional first user prompt.")
    parser.add_argument("--system", help="Optional system prompt.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct", help="HF model name.")
    parser.add_argument("--device", default="auto", help="Torch device. Use 'auto' to pick a supported accelerator or CPU.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Model weight dtype. 'auto' uses float32 on CPU and the checkpoint dtype on GPU. "
        "On a memory-constrained GPU pass 'float16' to roughly halve VRAM use.",
    )
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens generated for each assistant reply.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p (only used when temperature > 0).")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--native-preprocessing",
        action="store_true",
        help="Opt out of attack-matched preprocessing and let the processor handle the raw image "
        "(original resolution, default rescale) -- i.e. the realistic 'victim just opens the file' "
        "pipeline. By default the image is preprocessed exactly like scripts/run_poc.py so the prompt "
        "is the only difference between attack and chat.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=448,
        help="Square resize applied in the default attack-matched preprocessing. Must match "
        "run_poc.py's --resize (default 448) for the two pipelines to be identical.",
    )
    return parser.parse_args()


def build_user_message(text: str, include_image: bool) -> dict[str, object]:
    content: list[dict[str, str]] = []
    if include_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def build_messages(
    system_prompt: str | None,
    turns: list[dict[str, object]],
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.extend(turns)
    return messages


def prepare_image_input(image: Image.Image, native: bool, resize: int) -> tuple[object, bool]:
    """Return ``(processor_image, do_rescale)`` for the processor call.

    By default (attack-matched) the image goes through the exact same pixel transform as
    ``run_poc.py``: ``image_to_tensor`` (CHW float in [0, 1]) then ``prepare_image_batch``
    (clamp + bilinear resize to ``resize`` x ``resize``). The processor's ``do_rescale`` is
    therefore turned off, since the tensor is already in [0, 1]. The two pipelines then share
    the same code path, so an image is preprocessed identically in attack and chat.

    With ``native=True`` the raw PIL image is handed to the processor with its default
    rescaling and smart-resize -- the realistic pipeline a victim would use.
    """
    if native:
        return image, True

    tensor = image_to_tensor(image)
    processor_image = prepare_image_batch(tensor.unsqueeze(0), resize)[0]
    return processor_image, False


def generate_reply(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    processor_image: object,
    messages: list[dict[str, object]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_rescale: bool,
) -> str:
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        images=[processor_image],
        text=[prompt_text],
        return_tensors="pt",
        padding=True,
        do_rescale=do_rescale,
    )
    inputs.pop("token_type_ids", None)
    model_inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    error: list[BaseException] = []

    def _run() -> None:
        try:
            with torch.inference_mode():
                model.generate(**model_inputs, streamer=streamer, **generation_kwargs)
        except BaseException as exc:  # surfaced to the main thread after join
            error.append(exc)
            streamer.end()

    thread = Thread(target=_run, daemon=True)
    thread.start()

    print("Assistant: ", end="", flush=True)
    pieces: list[str] = []
    for chunk in streamer:
        print(chunk, end="", flush=True)
        pieces.append(chunk)
    thread.join()
    print("\n")
    if error:
        raise error[0]
    return "".join(pieces).strip()


def read_line(prompt_text: str) -> str | None:
    try:
        return input(prompt_text)
    except EOFError:
        return None


def main() -> None:
    args = parse_args()

    resolved_device = _resolve_device(args.device)
    resolved_dtype = _resolve_torch_dtype(args.dtype, resolved_device)
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
        low_cpu_mem_usage=True,
    )
    model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)  # inference only; no need for parameter gradients

    if args.seed is not None:
        torch.manual_seed(args.seed)

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    system_prompt = args.system
    turns: list[dict[str, object]] = []
    include_image = True

    def show_info() -> None:
        mode = (
            "native (raw image, processor default)"
            if args.native_preprocessing
            else f"attack-matched (resize={args.resize}, do_rescale=off)"
        )
        decoding = "greedy" if args.temperature <= 0.0 else f"sampling T={args.temperature} top_p={args.top_p}"
        print(
            f"model={args.model_name} device={model.device} dtype={next(model.parameters()).dtype}\n"
            f"image={image_path} ({image.size[0]}x{image.size[1]}) preprocessing={mode}\n"
            f"decoding={decoding} max_new_tokens={args.max_new_tokens} system={'set' if system_prompt else 'none'}"
        )

    print(f"Loaded {image_path} on {model.device}. Type /help for commands, /exit to quit.")
    show_info()

    pending = args.prompt
    while True:
        if pending is not None:
            user_text = pending.strip()
            pending = None
            if not user_text:
                continue
            print(f"You: {user_text}")
        else:
            line = read_line("You: ")
            if line is None:
                break
            user_text = line.strip()
            if not user_text:
                continue

        # ---- command handling -------------------------------------------------
        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if user_text.lower() == "/help":
            print(HELP_TEXT)
            continue
        if user_text.lower() == "/info":
            show_info()
            continue
        if user_text.lower() == "/reset":
            turns = []
            include_image = True
            print("Conversation cleared.\n")
            continue
        if user_text.lower().startswith("/system"):
            system_prompt = user_text[len("/system"):].strip() or None
            turns = []
            include_image = True
            print(f"System prompt {'set' if system_prompt else 'cleared'}; conversation reset.\n")
            continue
        if user_text.lower().startswith("/image"):
            new_path = user_text[len("/image"):].strip().strip('"')
            try:
                image = Image.open(Path(new_path)).convert("RGB")
            except (OSError, ValueError) as exc:
                print(f"Could not open image {new_path!r}: {exc}\n")
                continue
            image_path = Path(new_path)
            turns = []
            include_image = True
            free_memory(model.device)
            print(f"Loaded {image_path} ({image.size[0]}x{image.size[1]}); conversation reset.\n")
            continue
        if user_text.lower().startswith("/save"):
            out_path = user_text[len("/save"):].strip().strip('"') or "chat_transcript.txt"
            lines = []
            if system_prompt:
                lines.append(f"[system] {system_prompt}")
            for turn in turns:
                role = turn["role"]
                text = " ".join(part.get("text", "") for part in turn["content"] if part.get("type") == "text")
                lines.append(f"[{role}] {text}")
            Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Saved transcript to {out_path}\n")
            continue
        if user_text.lower() == "/regen":
            if len(turns) < 2 or turns[-1]["role"] != "assistant":
                print("Nothing to regenerate yet.\n")
                continue
            turns.pop()  # drop previous assistant reply; keep the last user turn
        else:
            turns.append(build_user_message(user_text, include_image=include_image))
            include_image = False

        # ---- generation -------------------------------------------------------
        processor_image, do_rescale = prepare_image_input(image, args.native_preprocessing, args.resize)
        messages = build_messages(system_prompt, turns)
        start = time.perf_counter()
        try:
            reply = generate_reply(
                model=model,
                processor=processor,
                processor_image=processor_image,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_rescale=do_rescale,
            )
        except KeyboardInterrupt:
            print("\n[generation interrupted]\n")
            # Roll back the user turn that produced no reply so the history stays consistent.
            if turns and turns[-1]["role"] == "user":
                turns.pop()
            continue
        elapsed = time.perf_counter() - start
        turns.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        print(f"[{elapsed:.1f}s]\n")


if __name__ == "__main__":
    main()
