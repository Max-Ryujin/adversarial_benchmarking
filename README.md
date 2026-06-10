# adversarial_benchmarking

Proof-of-concept code for attacking vision-language models through the logits of the first generated answer token.

Current scope:

- Qwen `Qwen/Qwen3-VL-4B-Instruct`
- Letter-based multiple-choice classification prompts
- PGD attacks on raw image pixels
- Local APGD variant (DLR loss, momentum, restarts) for untargeted and targeted first-token attacks

## Setup

```bash
pip install -e .
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

# Local APGD variant
python scripts/run_poc.py \
  --image path/to/image.jpg \
  --true-label cat \
  --target-label dog \
  --prompt "What animal is in the image?" \
  --choices cat dog horse bird \
  --attack apgd \
  --output-dir outputs/demo-apgd
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

## Performance and memory

The attacks run a full backward pass through a 4B-parameter VLM, which is heavy on both a
CPU and an older GPU. A few flags help:

- `--dtype float16` — on a memory-constrained GPU (e.g. an older AMD card) this roughly halves
  VRAM. `auto` (the default) keeps CPU in `float32` (needed for the backward pass) and uses the
  checkpoint dtype on a GPU. `float16`/`bfloat16` are ignored on CPU.
- `--grad-checkpointing` — trades ~30% more compute per step for a large drop in peak activation
  memory during the attack. Use this when you hit out-of-memory errors. The result is unchanged.

Other optimizations are always on and do not change behavior: weights are frozen
(`requires_grad_(False)`), the model loads with `low_cpu_mem_usage=True`, evaluation forwards
run under `inference_mode`, the KV cache is disabled for the single-step forwards, and the
expensive per-step diagnostics are only computed under `--debug` (they otherwise forced a
host/device sync every step).

Example, memory-constrained GPU:

```bash
python scripts/run_poc.py \
  --image path/to/image.jpg --true-label cat \
  --prompt "What animal is in the image?" --choices cat dog horse bird \
  --attack apgd --dtype float16 --grad-checkpointing \
  --output-dir outputs/demo-apgd
```

## Interactive image chat

```bash
python scripts/chat_with_image.py path/to/image.jpg \
  --prompt "Describe what you see in this image."
```

The script sends the image with the first user turn, streams the assistant reply token by
token, and keeps reading follow-up prompts until you type `/exit`. In-chat commands:

- `/help`, `/info` — usage and current settings
- `/reset` — clear the conversation, keep the image
- `/image <path>` — load a different image and start fresh
- `/system <text>` — set a system prompt and start fresh
- `/regen` — regenerate the last reply
- `/save <path>` — write the transcript to a file
- `/exit`, `/quit`

It accepts the same `--dtype`, `--resize`, `--min-pixels`, and `--max-pixels` options as the
attack script, plus `--system`, `--top-p`, `--seed`, and `--native-preprocessing`.

### Testing an adversarial image fairly

The chat script preprocesses the image through the **exact same code path** as `run_poc.py` by
default — `image_to_tensor` then `prepare_image_batch` (clamp + bilinear resize to `--resize`,
with the processor's rescale turned off). Both scripts share these two functions, so the image
the model sees is byte-identical and the **prompt is the only difference** between attack and
chat:

```bash
python scripts/chat_with_image.py outputs/demo/adversarial.png \
  --prompt "Choose exactly one option and answer with only the letter.\nA: cat\nB: dog"
```

Keep `--resize` (and any `--min-pixels`/`--max-pixels`) the same as the attack run for the
pipelines to match. To instead test the realistic "victim just opens the file" pipeline
(original resolution, processor default rescale/smart-resize), pass `--native-preprocessing`.

One residual difference is unavoidable: the attack optimizes the in-memory float tensor, while
the chat loads the 8-bit `adversarial.png`, so the saved image is the quantized version. If the
attack flips the answer but the matched-preprocessing chat does not even with the *same* prompt,
that gap is the `uint8` rounding (see below), not preprocessing.

See [docs/adversarial_transfer.md](docs/adversarial_transfer.md) for why an adversarial image
rarely transfers across prompts and how to craft one that does (EOT, quantization-aware steps,
prompt ensembling).

Targeted attack example:

```bash
python scripts/run_poc.py \
  --image path/to/image.jpg \
  --true-label real \
  --target-label ai-generated \
  --prompt "Is this image real or AI-generated?" \
  --choices real ai-generated \
  --attack apgd \
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
- `attacks/`: add additional gradient-based attacks
- `data/`: add dataset loaders for batch evaluation later
