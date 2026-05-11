# adversarial_benchmarking

Proof-of-concept code for attacking vision-language models through the logits of the first generated answer token.

Current scope:

- Qwen `Qwen/Qwen3-VL-4B-Instruct`
- Letter-based multiple-choice classification prompts
- PGD attacks on raw image pixels
- A thin adapter for future AutoAttack integration

## Setup

```bash
pip install -e .
```

Optional AutoAttack support:

```bash
pip install -e .[autoattack]
```

## How it works

The model is prompted with an image and a multiple-choice question where valid answers are single-token letters like `A`, `B`, `C`, or `D`.

Instead of using generated text directly as the attack target, the code reads the vocabulary logits at the final prompt position. Those logits correspond to the first generated answer token. It then slices those vocabulary logits down to the allowed answer letters and treats them as class logits.

## Example

```bash
python scripts/run_poc.py \
  --image path/to/image.jpg \
  --true-label cat \
  --prompt "What animal is in the image?" \
  --choices cat dog horse bird \
  --output-dir outputs/demo
```

Enable verbose tracing for the full pipeline:

```bash
python scripts/run_poc.py \
  --image path/to/image.jpg \
  --true-label cat \
  --prompt "What animal is in the image?" \
  --choices cat dog horse bird \
  --output-dir outputs/demo \
  --debug
```

Targeted attack example:

```bash
python scripts/run_poc.py \
  --image path/to/image.jpg \
  --true-label real \
  --target-label ai-generated \
  --prompt "Is this image real or AI-generated?" \
  --choices real ai-generated \
  --output-dir outputs/real-vs-ai
```

Artifacts are written to the output directory:

- `clean.png`
- `adversarial.png`
- `summary.json`
- `run.log` (`INFO` by default, full step-by-step traces with `--debug`)

## Extension points

- `tasks/`: add new prompt formats or other task definitions
- `models/`: add wrappers for other VLMs
- `attacks/`: add AutoAttack runs or additional gradient-based attacks
- `data/`: add dataset loaders for batch evaluation later
