from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from adversarial_benchmarking.attacks.apgd import apgd_attack
from adversarial_benchmarking.attacks.pgd import pgd_attack
from adversarial_benchmarking.config import AttackConfig, RunConfig
from adversarial_benchmarking.data.image_folder import load_image_tensor
from adversarial_benchmarking.logging_utils import configure_logging, get_logger
from adversarial_benchmarking.models.qwen3_vl import Qwen3VLFirstTokenClassifier, free_memory
from adversarial_benchmarking.tasks.first_token import FirstTokenTask
from adversarial_benchmarking.utils import save_image_tensor, write_json


logger = get_logger("scripts.jailbreak_poc")


def run_attack(
    model: Qwen3VLFirstTokenClassifier,
    clean_image: torch.Tensor,
    labels: torch.Tensor,
    attack_config: AttackConfig,
) -> torch.Tensor:
    if attack_config.name == "pgd":
        return pgd_attack(model, clean_image, labels, attack_config)
    if attack_config.name == "apgd":
        return apgd_attack(
            model=model,
            images=clean_image,
            labels=labels,
            config=attack_config,
            use_dlr=False,  # targeted objective uses -CE; DLR (untargeted) is irrelevant here.
            momentum=0.75,
            eot_iter=1,
        )
    raise ValueError(f"Unsupported attack: {attack_config.name}")


def _first_token_report(
    model: Qwen3VLFirstTokenClassifier, image: torch.Tensor, target_token_id: int
) -> tuple[float, int, str]:
    """Return (target_prob, argmax_token_id, argmax_token_text) for the first answer token."""
    result = model.predict_result(image)
    vocab_logits = result.vocab_logits[0]
    probs = F.softmax(vocab_logits.float(), dim=-1)
    target_prob = probs[target_token_id].item()
    argmax_id = int(vocab_logits.argmax().item())
    argmax_text = model.processor.tokenizer.decode([argmax_id])
    return target_prob, argmax_id, argmax_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jailbreak proof of concept: run a TARGETED first-token attack on the image so "
        "the model opens its reply with an affirmative token (e.g. 'Sure') and continues from there.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", required=True, help="The (potentially refused) question.")
    parser.add_argument(
        "--target",
        default="Sure",
        help="Target opening token. Only its FIRST token is attacked, so pick a word whose first "
        "token already commits the model to comply (e.g. 'Sure', 'Yes', 'Certainly').",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional forced assistant prefix injected during BOTH attack and generation. The "
        "attack then targets the first token AFTER this prefix. Empty by default.",
    )
    parser.add_argument("--output-dir", default="outputs/jailbreak", help="Directory for artifacts.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct", help="HF model name.")
    parser.add_argument("--device", default="auto", help="Torch device ('auto' picks an accelerator or CPU).")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Model weight dtype. 'auto' uses float32 on CPU and the checkpoint dtype on GPU.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="Trade ~30%% more compute per step for much lower peak memory. Use on OOM.",
    )
    parser.add_argument("--resize", type=int, default=448, help="Fixed square resize before the processor.")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument(
        "--attack",
        choices=("pgd", "apgd"),
        default="apgd",
        help="Attack implementation. 'apgd' is the local adaptive variant.",
    )
    parser.add_argument("--norm", default="Linf", help="Threat-model norm (Linf only).")
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=2.0 / 255.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Tokens generated for reporting.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_config = RunConfig(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        true_label="",
        prompt=args.prompt,
        choices=[],
        model_name=args.model_name,
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        resize=args.resize,
        target_label=args.target,
        dtype=args.dtype,
        grad_checkpointing=args.grad_checkpointing,
        debug=args.debug,
    )
    attack_config = AttackConfig(
        name=args.attack,
        norm=args.norm,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        targeted=True,  # a jailbreak always pushes TOWARD an affirmative token.
        random_start=not args.no_random_start,
    )

    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(debug=run_config.debug, log_file=run_config.output_dir / "run.log")
    logger.info("Starting jailbreak run")
    logger.debug("Run config: %s", asdict(run_config))
    logger.debug("Attack config: %s", asdict(attack_config))

    processor = AutoProcessor.from_pretrained(run_config.model_name)
    task = FirstTokenTask.from_target(
        instruction=run_config.prompt,
        target_text=args.target,
        tokenizer=processor.tokenizer,
    )

    model = Qwen3VLFirstTokenClassifier(
        model_name=run_config.model_name,
        task=task,
        device=run_config.device,
        min_pixels=run_config.min_pixels,
        max_pixels=run_config.max_pixels,
        resize=run_config.resize,
        torch_dtype=args.dtype,
        grad_checkpointing=args.grad_checkpointing,
        assistant_prefix=args.prefix,
    )

    clean_image = load_image_tensor(run_config.image_path).unsqueeze(0).to(model.device)
    labels = torch.tensor([task.target_token_id], device=model.device)

    clean_prob, clean_top_id, clean_top_text = _first_token_report(model, clean_image, task.target_token_id)
    clean_generation = model.generate_text(clean_image, max_new_tokens=args.max_new_tokens)[0]
    logger.info(
        "Clean: P(target=%r)=%.4f, model would open with %r",
        args.target,
        clean_prob,
        clean_top_text,
    )

    adv_image = run_attack(model, clean_image, labels, attack_config)
    free_memory(model.device)

    adv_prob, adv_top_id, adv_top_text = _first_token_report(model, adv_image, task.target_token_id)
    adv_generation = model.generate_text(adv_image, max_new_tokens=args.max_new_tokens)[0]
    logger.info(
        "Adversarial: P(target=%r)=%.4f, model opens with %r",
        args.target,
        adv_prob,
        adv_top_text,
    )

    save_image_tensor(clean_image[0], run_config.output_dir / "clean.png")
    save_image_tensor(adv_image[0], run_config.output_dir / "adversarial.png")

    summary = {
        "prompt": run_config.prompt,
        "assistant_prefix": args.prefix,
        "target_text": args.target,
        "target_token_id": task.target_token_id,
        "clean_target_prob": clean_prob,
        "adv_target_prob": adv_prob,
        "clean_first_token": clean_top_text,
        "adv_first_token": adv_top_text,
        "clean_generation": args.prefix + clean_generation,
        "adv_generation": args.prefix + adv_generation,
        "attack": {
            "name": attack_config.name,
            "norm": attack_config.norm,
            "epsilon": attack_config.epsilon,
            "step_size": attack_config.step_size,
            "steps": attack_config.steps,
            "targeted": attack_config.targeted,
            "random_start": attack_config.random_start,
        },
    }
    write_json(summary, run_config.output_dir / "summary.json")
    logger.info("Artifacts written to %s", run_config.output_dir)

    print("\n=== CLEAN ===")
    print(f"P(target {args.target!r}) = {clean_prob:.4f}  |  first token: {clean_top_text!r}")
    print(f"{args.prefix}{clean_generation}")
    print("\n=== ADVERSARIAL ===")
    print(f"P(target {args.target!r}) = {adv_prob:.4f}  |  first token: {adv_top_text!r}")
    print(f"{args.prefix}{adv_generation}")


if __name__ == "__main__":
    main()
