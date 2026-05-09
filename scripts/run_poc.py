from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor

from adversarial_benchmarking.attacks.pgd import pgd_attack
from adversarial_benchmarking.config import AttackConfig, RunConfig
from adversarial_benchmarking.data.image_folder import load_image_tensor
from adversarial_benchmarking.models.qwen3_vl import Qwen3VLFirstTokenClassifier
from adversarial_benchmarking.tasks.multiple_choice import MultipleChoiceTask
from adversarial_benchmarking.utils import save_image_tensor, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qwen3-VL adversarial proof of concept.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--true-label", required=True, help="Semantic label for the clean image.")
    parser.add_argument("--prompt", required=True, help="Task instruction shown to the model.")
    parser.add_argument("--choices", required=True, nargs="+", help="Semantic labels for A/B/C/... options.")
    parser.add_argument("--target-label", help="Optional target label for targeted PGD.")
    parser.add_argument("--output-dir", default="outputs/poc", help="Directory for artifacts.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-4B-Instruct", help="HF model name.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--resize", type=int, default=448, help="Fixed square resize before processor.")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=2.0 / 255.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--no-random-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_config = RunConfig(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        true_label=args.true_label,
        prompt=args.prompt,
        choices=args.choices,
        model_name=args.model_name,
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        resize=args.resize,
        target_label=args.target_label,
    )
    attack_config = AttackConfig(
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        targeted=args.target_label is not None,
        random_start=not args.no_random_start,
    )

    run_config.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(run_config.model_name)
    task = MultipleChoiceTask.from_labels(
        instruction=run_config.prompt,
        labels=run_config.choices,
        tokenizer=processor.tokenizer,
    )
    task.class_index_for_label(run_config.true_label)
    if run_config.target_label is not None:
        task.class_index_for_label(run_config.target_label)

    model = Qwen3VLFirstTokenClassifier(
        model_name=run_config.model_name,
        task=task,
        device=run_config.device,
        min_pixels=run_config.min_pixels,
        max_pixels=run_config.max_pixels,
        resize=run_config.resize,
    )

    clean_image = load_image_tensor(run_config.image_path).unsqueeze(0).to(model.device)
    label_name = run_config.target_label if attack_config.targeted else run_config.true_label
    label_index = task.class_index_for_label(label_name)
    labels = torch.tensor([label_index], device=model.device)

    clean_result = model.forward_result(clean_image)
    clean_prediction_index = clean_result.class_logits.argmax(dim=-1).item()
    clean_generation = model.generate_letters(clean_image)[0]

    adv_image = pgd_attack(model, clean_image, labels, attack_config)
    adv_result = model.forward_result(adv_image)
    adv_prediction_index = adv_result.class_logits.argmax(dim=-1).item()
    adv_generation = model.generate_letters(adv_image)[0]

    save_image_tensor(clean_image[0], run_config.output_dir / "clean.png")
    save_image_tensor(adv_image[0], run_config.output_dir / "adversarial.png")

    summary = {
        "prompt": task.build_prompt(),
        "choice_letters": [option.letter for option in task.options],
        "choices": task.class_labels(),
        "true_label": run_config.true_label,
        "target_label": run_config.target_label,
        "clean_prediction": task.class_labels()[clean_prediction_index],
        "adv_prediction": task.class_labels()[adv_prediction_index],
        "clean_generation": clean_generation,
        "adv_generation": adv_generation,
        "clean_logits": clean_result.class_logits[0].detach().cpu().tolist(),
        "adv_logits": adv_result.class_logits[0].detach().cpu().tolist(),
        "attack": {
            "epsilon": attack_config.epsilon,
            "step_size": attack_config.step_size,
            "steps": attack_config.steps,
            "targeted": attack_config.targeted,
            "random_start": attack_config.random_start,
        },
    }
    write_json(summary, run_config.output_dir / "summary.json")

    print(summary)


if __name__ == "__main__":
    main()
